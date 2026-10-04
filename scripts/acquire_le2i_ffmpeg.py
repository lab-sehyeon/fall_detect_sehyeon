"""Pinned, project-local FFmpeg tools; no package installation or PATH changes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/source_archives/Le2i-FDD/ffmpeg_20261002_r1'
URL = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-01-13-06/ffmpeg-n8.1.3-14-g330caae0c1-linux64-lgpl-8.1.tar.xz'
SIZE = 137039192
SHA256 = '3c2c4d6066432b2eab54be830d3ce1f5b68a3f34ecaa6d9f2d0230a49e778560'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def main():
    assert not any(p.is_symlink() for p in (DEST, *DEST.parents))
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / 'release.tar.xz'
    if not archive.exists():
        part = DEST / 'release.tar.xz.partial'
        assert not part.exists(), 'preserve unexpected partial'
        with urllib.request.urlopen(URL, timeout=90) as response, part.open('xb') as out:
            count = 0
            while True:
                block = response.read(1024**2)
                if not block: break
                count += len(block)
                assert count <= SIZE and shutil.disk_usage(ROOT).free > 64 * 1024**3
                out.write(block)
        assert part.stat().st_size == SIZE and sha(part) == SHA256
        part.rename(archive)
    assert archive.stat().st_size == SIZE and sha(archive) == SHA256
    extracted = {}
    with tarfile.open(archive) as bundle:
        for entry in bundle:
            path = PurePosixPath(entry.name)
            assert not path.is_absolute() and '..' not in path.parts
            name = path.name
            if name not in ('ffmpeg', 'ffprobe') and not name.upper().startswith(('LICENSE', 'COPYING')):
                continue
            assert entry.isfile() and entry.size < 400 * 1024**2 and name not in extracted
            target = DEST / name
            assert not target.is_symlink()
            if not target.exists():
                with bundle.extractfile(entry) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, 1024**2)
            assert target.stat().st_size == entry.size
            if name in ('ffmpeg', 'ffprobe'): target.chmod(0o755)
            extracted[name] = {'bytes': entry.size, 'sha256': sha(target)}
    assert {'ffmpeg', 'ffprobe'}.issubset(extracted)
    versions = {name: subprocess.check_output([str(DEST/name), '-version'], text=True).splitlines()[0]
                for name in ('ffmpeg', 'ffprobe')}
    result = dict(url=URL, archive_sha256=SHA256, archive_bytes=SIZE, files=extracted, versions=versions,
                  official_link='https://ffmpeg.org/download.html', no_system_install=True)
    manifest = DEST / 'manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == result
    else: manifest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
