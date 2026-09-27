"""Storage recovery CLI: same durable operations as System > Storage."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import media_storage as storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'enable', 'preview', 'copy', 'cleanup', 'pause', 'resume', 'cancel', 'run'])
    parser.add_argument('--yes', action='store_true', help='Confirm starting migration or deleting verified local copies')
    args = parser.parse_args()
    if args.action in {'copy', 'cleanup'}:
        if not args.yes:
            parser.error('Review preview/status first, then supply --yes')
        storage.start(args.action)
    elif args.action == 'enable':
        storage.enable()
    elif args.action == 'preview':
        print(json.dumps(storage.preview(), indent=2)); return
    elif args.action in {'pause', 'resume', 'cancel'}:
        storage.control(args.action)
    elif args.action == 'run':
        while storage.state()['status'] in {'waiting', 'copying', 'cleaning', 'cancelling'}:
            storage.step(); time.sleep(0.2)
    print(json.dumps(storage.status(), indent=2))
    if storage.state()['status'] == 'error':
        sys.exit(1)


if __name__ == '__main__':
    main()
