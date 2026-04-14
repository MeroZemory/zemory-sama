import asyncio
import sys

from zemory.orchestrator import run


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
