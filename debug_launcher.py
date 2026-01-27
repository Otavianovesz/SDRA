import logging
import sys
import os
from datetime import datetime

# Generate a log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"debug_log_{timestamp}.txt"

# Configure logging BEFORE importing main variables to ensure we capture everything
# We use force=True to overwrite any previous configuration if it exists (though not expected at start)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logging.info(f"Iniciando modo de depuração. Log sendo salvo em: {os.path.abspath(log_filename)}")

try:
    # Adding current directory to path just in case
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import the application from main.py
    # NOTE: main.py has a logging.basicConfig call. Since we configured it before with force=True, 
    # or if main.py runs after, we need to be careful. 
    # Actually, logging.basicConfig is a no-op if the root logger already has handlers, 
    # unless force=True is used. main.py probably doesn't use force=True.
    from main import SRDAApplication
    
    # Initialize and run
    logging.info("Instanciando SRDAApplication...")
    app = SRDAApplication()
    logging.info("Iniciando mainloop...")
    app.run()
    
except Exception as e:
    logging.critical(f"CRASH não tratado: {e}", exc_info=True)
    input("Pressione Enter para sair...")
