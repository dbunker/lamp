
import logging
import os
import time

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

def run_commands():
    name = 'paper'
    os.system('bibtex ' + name)
    os.system('pdflatex -interaction=nonstopmode ' + name)

class SphinxEventHandler(PatternMatchingEventHandler):
    """Rebuild and refresh on every change event."""

    def __init__(self, patterns=None, ignore_patterns=None,
            ignore_directories=False, case_sensitive=False):

        super(SphinxEventHandler, self).__init__(patterns=patterns, ignore_patterns=ignore_patterns,
                ignore_directories=ignore_directories, case_sensitive=case_sensitive)

    def on_modified(self, event):
        super(SphinxEventHandler, self).on_modified(event)

        run_commands()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    event_handler = SphinxEventHandler(patterns=['*.bib', '*.tex', '*.cls', '.sty', '.bst'])
    observer = Observer()
    observer.schedule(event_handler, path=os.path.abspath("."),
            recursive=True)
    observer.start()

    run_commands()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    event_handler.cleanup()
    observer.join()
