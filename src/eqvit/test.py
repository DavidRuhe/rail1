import logging

# Configure global logging
logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(filename)s:%(lineno)d (%(levelname)s) %(message)s')

# Example usage with global logging
logging.debug("This is a global debug message")
logging.info("This is a global info message")
logging.warning("This is a global warning message")
logging.error("This is a global error message")
logging.critical("This is a global critical message")

