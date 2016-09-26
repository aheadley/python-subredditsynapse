#!/usr/bin/env python2

from codecs import open
from six.moves.html_parser import HTMLParser

from subredditsynapse.data import RedditDataDump

HTML_PARSER = HTMLParser()

if __name__ == '__main__':
    import sys

    dd = RedditDataDump(sys.argv[1],
        transform_func=lambda c: HTML_PARSER.unescape(c['body']).decode('utf-8'))

    limit = 100000
    b = 0
    with open(sys.argv[2], 'wb', encoding='utf-8') as out_f:
        while b < limit:
            c = next(dd)
            b += len(c)
            out_f.write(c)
            # out_f.write('\x00')
