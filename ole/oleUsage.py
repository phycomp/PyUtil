  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -m, --man             Print manual
  -s SELECT, --select=SELECT
                        select item nr for dumping (a for all)
  -d, --dump            perform dump
  -x, --hexdump         perform hex dump
  -a, --asciidump       perform ascii dump
  -A, --asciidumprle    perform ascii dump with RLE
  -S, --strings         perform strings dump
  -T, --headtail        do head & tail
  -v, --vbadecompress   VBA decompression
  --vbadecompressskipattributes
                        VBA decompression, skipping initial attributes
  --vbadecompresscorrupt
                        VBA decompression, display beginning if corrupted
  -r, --raw             read raw file (use with options -v or -p
  -t TRANSLATE, --translate=TRANSLATE
                        string translation, like utf16 or .decode("utf8")
  -e, --extract         extract OLE embedded file
  -i, --info            print extra info for selected item
  -p PLUGINS, --plugins=PLUGINS
                        plugins to load (separate plugins with a comma , ;
                        @file supported)
  --pluginoptions=PLUGINOPTIONS
                        options for the plugin
  --plugindir=PLUGINDIR
                        directory for the plugin
  -q, --quiet           only print output from plugins
  -y YARA, --yara=YARA  YARA rule-file, @file, directory or #rule to check
                        streams (YARA search doesn't work with -s option)
  -D DECODERS, --decoders=DECODERS
                        decoders to load (separate decoders with a comma , ;
                        @file supported)
  --decoderoptions=DECODEROPTIONS
                        options for the decoder
  --decoderdir=DECODERDIR
                        directory for the decoder
  --yarastrings         Print YARA strings
  -M, --metadata        Print metadata
  -c, --calc            Add extra calculated data to output, like hashes
  --decompress          Search for compressed data in the stream and
                        decompress it
  -V, --verbose         verbose output with decoder errors and YARA rules
  -C CUT, --cut=CUT     cut data
  -E EXTRA, --extra=EXTRA
                        add extra info (environment variable: OLEDUMP_EXTRA)
  --storages            Include storages in report
  -f FIND, --find=FIND  Find D0CF11E0 MAGIC sequence (use l for listing,
                        number for selecting)
  -j, --jsonoutput      produce json output
  -u, --unuseddata      Include unused data after end of stream
  --password=PASSWORD   The ZIP password to be used (default infected)
