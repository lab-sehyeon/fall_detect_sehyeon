"""Read-only movie/track header diagnosis of OOPS source-duration discrepancies.

Header fields checked against FFmpeg n7.1 libavformat/mov.c mov_read_mvhd/mdhd.
Not a full demuxer, edit-list audit, repair, or permission to alter annotation ends.
"""
import struct
from pathlib import Path

from scripts import prepare_rgb_recovery_data_r2 as prep


def boxes(stream, start, end):
    offset = start
    while offset < end:
        prep.require(end - offset >= 8, 'truncated atom header')
        stream.seek(offset)
        size, kind = struct.unpack('>I4s', stream.read(8))
        header = 8
        if size == 1:
            prep.require(end - offset >= 16, 'truncated extended atom header')
            size = struct.unpack('>Q', stream.read(8))[0]
            header = 16
        elif size == 0:
            size = end - offset
        prep.require(header <= size <= end - offset, 'atom outside parent bounds')
        yield kind, offset + header, offset + size
        offset += size


def duration_header(body):
    prep.require(len(body) >= 4 and body[0] in (0, 1), 'unsupported duration header')
    version = body[0]
    offset = 20 if version == 1 else 12  # version/flags + creation/modification times
    prep.require(len(body) >= offset + (12 if version else 8), 'truncated duration header')
    scale = struct.unpack_from('>I', body, offset)[0]
    duration = struct.unpack_from('>Q' if version else '>I', body, offset + 4)[0]
    prep.require(scale > 0 and duration not in (2**32 - 1, 2**64 - 1), 'unknown media duration')
    return dict(version=version, timescale=scale, duration_ticks=duration, seconds=duration / scale)


def movie_headers(path):
    with path.open('rb') as stream:
        movies = [b for b in boxes(stream, 0, path.stat().st_size) if b[0] == b'moov']
        prep.require(len(movies) == 1, 'expected one moov')
        children = list(boxes(stream, movies[0][1], movies[0][2]))
        headers = [b for b in children if b[0] == b'mvhd']
        prep.require(len(headers) == 1, 'expected one mvhd')
        stream.seek(headers[0][1])
        movie = duration_header(stream.read(min(40, headers[0][2] - headers[0][1])))
        tracks = []
        for _, start, end in (b for b in children if b[0] == b'trak'):
            media = [b for b in boxes(stream, start, end) if b[0] == b'mdia']
            prep.require(len(media) == 1, 'expected one mdia per track')
            fields = list(boxes(stream, media[0][1], media[0][2]))
            mdhd = [b for b in fields if b[0] == b'mdhd']
            handlers = [b for b in fields if b[0] == b'hdlr']
            prep.require(len(mdhd) == len(handlers) == 1, 'expected mdhd/hdlr')
            stream.seek(mdhd[0][1])
            duration = duration_header(stream.read(min(40, mdhd[0][2] - mdhd[0][1])))
            prep.require(handlers[0][2] - handlers[0][1] >= 12, 'truncated handler')
            stream.seek(handlers[0][1] + 8)
            handler = stream.read(4).decode('ascii')
            tracks.append(dict(handler=handler, media_duration=duration))
        return dict(movie_duration=movie, tracks=tracks)


def main():
    audit_path = prep.OUTPUT / 'oops_final_report.json'
    parent = prep.read(audit_path)
    pins = prep.read(prep.source.CONTROL / 'oops_videos.json')
    records = []
    for row in parent['source_timeline_mismatches']:
        prep.safety()
        path = prep.source.safe_path(prep.source.OOPS, row['video'])
        prep.require(prep.sha(path) == pins[row['video']]['sha256'], 'source changed')
        headers = movie_headers(path)
        movie_end = headers['movie_duration']['seconds']
        videos = [r['media_duration']['seconds'] for r in headers['tracks'] if r['handler'] == 'vide']
        prep.require(len(videos) == 1, 'expected one video track')
        records.append(dict(**row, headers=headers,
                            annotation_minus_movie_seconds=row['annotation_end_seconds'] - movie_end,
                            video_header_minus_decoded_seconds=videos[0] - row['duration_seconds'],
                            annotation_within_movie_1ms=row['annotation_end_seconds'] <= movie_end + .001))
    report = dict(passed=True, scope='header diagnosis only', videos=len(records), records=records,
                  parent_report_sha256=prep.sha(audit_path), code_sha256=prep.sha(Path(__file__)),
                  parent_source_timeline_gate_overridden=False,
                  all_annotation_ends_within_movie_1ms=all(r['annotation_within_movie_1ms'] for r in records),
                  full_pts_and_edit_lists_audited=False, source_or_labels_modified=False,
                  model_evaluation_ready=False)
    prep.save(prep.OUTPUT / 'oops_container_timing_report.json', report)
    print({k: v for k, v in report.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    main()
