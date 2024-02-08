import logging

# Configure global logging
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d (%(levelname)s) %(message)s",
)
