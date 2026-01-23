"""
MCMF Reconciler using OR-Tools
==============================
Algorithmic Reconciliation using Constraint Programming (CP-SAT).

Solves the N-to-M matching problem:
- Matches Payments to Invoices (Boletos/NFes)
- Maximizes total matched value (Primary Objective)
- Minimizes matching cost/similarity (Secondary Objective)
- Supports partial matches and many-to-many relationships

Mathematical Model:
    Variables: x[p, i] (int) = Amount in cents matched between Payment p and Invoice i
    
    maximize: (Sum(x[p,i]) * WEIGHT_VALUE) - Sum(x[p,i] * Cost[p,i])
    
    subject to:
        Sum(x[p, :]) <= Payment[p].amount
        Sum(x[:, i]) <= Invoice[i].amount
        x[p, i] >= 0
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, date

from ortools.sat.python import cp_model


class DocumentNode:
    """Node representing a document in the reconciliation graph."""
    def __init__(self, id: int, doc_type: str, amount_cents: int, supplier: Optional[str] = None,
                 due_date: Optional[str] = None, payment_date: Optional[str] = None,
                 emission_date: Optional[str] = None, entity_tag: Optional[str] = None,
                 payment_status: str = "UNKNOWN", original_path: str = ""):
        self.id = id
        self.doc_type = doc_type
        self.amount_cents = amount_cents
        self.supplier = supplier
        self.due_date = due_date
        self.payment_date = payment_date
        self.emission_date = emission_date
        self.entity_tag = entity_tag
        self.payment_status = payment_status
        self.original_path = original_path

@dataclass
class TransactionIsland:
    """Group of matched documents (e.g. 1 Boleto + 1 Comprovante)."""
    nodes: List[DocumentNode]
    edges: List = field(default_factory=list)
    total_value: int = 0
    is_complete: bool = False
    master_date: Optional[str] = None

logger = logging.getLogger(__name__)


class MatchingCostCalculator:
    """Calculates cost (0.0 to 1.0) of matching two documents."""
    
    @staticmethod
    def calculate_cost(payment: DocumentNode, invoice: DocumentNode) -> float:
        """
        Calculate matching cost (lower is better).
        
        Factors:
        1. Date difference (Payment Date vs Due Date)
        2. Supplier name similarity
        3. Entity tag consistency
        """
        cost = 0.0
        
        # 1. Date Diff (Weight: 0.5)
        # Ideal: Payment on or before Due Date, but close
        p_date = payment.payment_date or payment.due_date
        i_date = invoice.due_date or invoice.emission_date
        
        if p_date and i_date:
            try:
                d1 = datetime.strptime(p_date, "%Y-%m-%d")
                d2 = datetime.strptime(i_date, "%Y-%m-%d")
                diff = abs((d1 - d2).days)
                
                # Penalty for huge gaps
                if diff > 45: 
                    cost += 0.5
                else:
                    cost += (diff / 45.0) * 0.3
            except:
                cost += 0.2
        else:
            cost += 0.2
            
        # 2. Supplier Similarity (Weight: 0.3)
        if payment.supplier and invoice.supplier:
            # Simple exact match check for speed, else normalized
            s1 = payment.supplier.upper().strip()
            s2 = invoice.supplier.upper().strip()
            
            if s1 == s2:
                cost += 0.0
            elif s1 in s2 or s2 in s1:
                cost += 0.1
            else:
                cost += 0.3  # Supplier mismatch penalty
        
        # 3. Entity Tag (Weight: 0.2)
        if payment.entity_tag and invoice.entity_tag:
            if payment.entity_tag != invoice.entity_tag:
                cost += 1.0  # Strong penalty for wrong entity (VG vs MV)
        
        return min(cost, 1.0)


class MCMFReconciler:
    """
    Min-Cost Max-Flow Reconciler using Google OR-Tools CP-SAT.
    """
    
    def __init__(self, db=None):
        self.db = db
        self.cost_calculator = MatchingCostCalculator()
        
    def reconcile(self, documents: List[DocumentNode]) -> List[TransactionIsland]:
        """
        Run global reconciliation on provided documents.
        
        Args:
            documents: List of all relevant documents
            
        Returns:
            List of 'Islands' (matched groups)
        """
        if not documents:
            return []
            
        # 1. Separate into Payments and Invoices
        payments = []
        invoices = []
        
        for doc in documents:
            if doc.doc_type == "COMPROVANTE":
                payments.append(doc)
            elif doc.doc_type in ["BOLETO", "NFE", "NFSE", "FATURA"]:
                invoices.append(doc)
            else:
                # Treat UNKNOWN as Invoice (conservative)
                invoices.append(doc)
                
        logger.info(f"Reconciling {len(payments)} payments vs {len(invoices)} invoices")
        
        # 2. Build CP-SAT Model
        model = cp_model.CpModel()
        
        # Decision Variables: x[p, i] = amount matched
        match_vars = {}
        possible_matches = []
        
        # Heuristic: Only create variables for plausible matches to save RAM
        # i.e. Dates within 60 days
        valid_pairs_count = 0
        
        for p_idx, p in enumerate(payments):
            for i_idx, i in enumerate(invoices):
                # Basic feasibility filter
                # If entity tags differ, skip (hard constraint simulation)
                if p.entity_tag and i.entity_tag and p.entity_tag != i.entity_tag:
                    continue
                    
                # Variable creation
                # Name: x_p_i
                var = model.NewIntVar(0, min(p.amount_cents, i.amount_cents), f"x_{p.id}_{i.id}")
                match_vars[(p_idx, i_idx)] = var
                
                # Calculate cost (0-100 ints for solver)
                cost_float = self.cost_calculator.calculate_cost(p, i)
                cost_int = int(cost_float * 100)
                
                possible_matches.append({
                    "var": var,
                    "p_idx": p_idx,
                    "i_idx": i_idx,
                    "cost": cost_int
                })
                valid_pairs_count += 1

        logger.info(f"Created {valid_pairs_count} decision variables")
        
        # 3. Constraints
        
        # Sum of matches for each payment <= payment amount
        for p_idx, p in enumerate(payments):
            p_matches = [m['var'] for m in possible_matches if m['p_idx'] == p_idx]
            if p_matches:
                model.Add(sum(p_matches) <= p.amount_cents)
                
        # Sum of matches for each invoice <= invoice amount
        for i_idx, i in enumerate(invoices):
            i_matches = [m['var'] for m in possible_matches if m['i_idx'] == i_idx]
            if i_matches:
                model.Add(sum(i_matches) <= i.amount_cents)
        
        # 4. Objective Function
        # Maximize: (Matched Amount * 1000) - (Matched Amount * Cost)
        # This creates a "Min-Cost Max-Flow" equivalent: 
        # Priority 1: Maximize Amount
        # Priority 2: Minimize Cost (prefer matches with lower cost)
        
        objective_terms = []
        for match in possible_matches:
            # Profit = 1000 - Cost (so minimization becomes maximization)
            # Cost is 0-100. Profit range 900-1000.
            profit = 1000 - match['cost']
            objective_terms.append(match['var'] * profit)
            
        if objective_terms:
            model.Maximize(sum(objective_terms))
            
        # 5. Solve
        solver = cp_model.CpSolver()
        # Time limit 30s
        solver.parameters.max_time_in_seconds = 30.0
        
        status = solver.Solve(model)
        
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.warning("No solution found by OR-Tools")
            return []
            
        logger.info(f"Solution found: {solver.StatusName(status)}")
        logger.info(f"Objective value: {solver.ObjectiveValue()}")
        
        # 6. Reconstruct Graphs (Islands)
        # We need to group connected components
        
        # Build adjacency list
        adj = {}
        
        for match in possible_matches:
            amt = solver.Value(match['var'])
            if amt > 0:
                p_id = payments[match['p_idx']].id
                i_id = invoices[match['i_idx']].id
                
                # Add edge p <-> i
                if p_id not in adj: adj[p_id] = []
                if i_id not in adj: adj[i_id] = []
                
                adj[p_id].append(i_id)
                adj[i_id].append(p_id)
        
        # Find connected components (BFS/DFS)
        visited = set()
        islands = []
        
        # Also include orphans (nodes with degree 0)
        all_node_ids = {d.id: d for d in documents}
        
        for node_id in all_node_ids:
            if node_id in visited:
                continue
                
            # Start new component
            component_nodes = []
            stack = [node_id]
            visited.add(node_id)
            
            while stack:
                curr = stack.pop()
                component_nodes.append(all_node_ids[curr])
                
                # Neighbors
                neighbors = adj.get(curr, [])
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        stack.append(n)
            

            # Create Island
            # Determine if it's 'complete' (Total Payments == Total Invoices)
            island_payments = sum(n.amount_cents for n in component_nodes if n.doc_type == "COMPROVANTE")
            island_invoices = sum(n.amount_cents for n in component_nodes if n.doc_type != "COMPROVANTE")
            
            is_complete = (island_payments == island_invoices) and (island_payments > 0)
            
            # Determine master date (logic from legacy)
            master_date = None
            dates = []
            for n in component_nodes:
                if n.payment_date: dates.append((3, n.payment_date))
                elif n.due_date: dates.append((2, n.due_date))
                elif n.emission_date: dates.append((1, n.emission_date))
            
            if dates:
                dates.sort(reverse=True) # Sort by priority
                master_date = dates[0][1]
            
            # Recapture edges for this island
            island_edges = []
            component_ids = {n.id for n in component_nodes}
            
            # Iterate through possible matches again (inefficient but safe) or usage adj
            # Use adj: it has both directions, be careful not to double count
            added_edges = set()
            for u in component_ids:
                if u in adj:
                    for v in adj[u]:
                        if v in component_ids:
                            # Verify if edge u-v was selected by solver (adj only contains selected)
                            edge_key = tuple(sorted((u, v)))
                            if edge_key not in added_edges:
                                island_edges.append({"source": u, "target": v, "weight": 1.0}) # Simplified weight
                                added_edges.add(edge_key)

            islands.append(TransactionIsland(
                nodes=component_nodes, 
                edges=island_edges,
                total_value=max(island_payments, island_invoices),
                is_complete=is_complete,
                master_date=master_date
            ))
            
        return islands
            
        return islands

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("MCMF Reconciler Test")
    
    # Mock data
    p1 = DocumentNode(1, "COMPROVANTE", 10000, "SUPPLIER A", payment_date="2025-01-10")
    i1 = DocumentNode(2, "BOLETO", 4000, "SUPPLIER A", due_date="2025-01-10")
    i2 = DocumentNode(3, "NFE", 6000, "SUPPLIER A", due_date="2025-01-10")
    i3 = DocumentNode(4, "BOLETO", 5000, "SUPPLIER B", due_date="2025-01-10") # Should act as orphan
    
    reconciler = MCMFReconciler()
    islands = reconciler.reconcile([p1, i1, i2, i3])
    
    print(f"Found {len(islands)} islands")
    for idx, island in enumerate(islands):
        print(f"Island {idx}: {len(island.nodes)} nodes, Value: {island.total_value}, Complete: {island.is_complete}")
        for n in island.nodes:
            print(f"  - {n.doc_type} #{n.id}: {n.amount_cents}")
    
    # Expected: 
    # Island with P1, I1, I2 (Complete, Value 10000)
    # Island with I3 (Orphan, Value 0 matched)
