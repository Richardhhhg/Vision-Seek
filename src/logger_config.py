import logging

ANSI = {
    "core": "\033[95m",
    "detect": "\033[92m",
}

RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    def format(self, record):
        color = ANSI.get(record.name, "")

        record.colored_name = f"{color}{record.name}{RESET}"
        record.colored_message = f"{color}{record.getMessage()}{RESET}"
        record.colored_levelname = f"{color}{record.levelname}{RESET}"

        return super().format(record)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColourFormatter("[%(colored_levelname)s] %(colored_name)s: %(colored_message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
