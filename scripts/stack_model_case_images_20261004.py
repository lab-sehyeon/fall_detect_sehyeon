"""Losslessly stack the five existing model PNGs for each fixed case."""
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'model_case_images_20261004'
OUT = ROOT.parent / 'case_stacked_images_20261004'
EVIDENCE = ROOT / 'docs/internal/2026-10-04_model_case_figures_evidence'
ORDER = ['01_Ours', '02_USDRL_NTU60', '03_HFD', '04_Privacy_X3D_UDA', '05_FLASH']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    prior = json.loads((EVIDENCE / 'validation.json').read_text())
    assert prior['passed'] and prior['images'] == 10
    outputs = ['Case1_Le2i_all_models_vertical.png', 'Case2_CAUCA_all_models_vertical.png']
    archive = OUT.with_suffix('.zip')
    assert not archive.exists(), 'Destination archive already exists'
    for name in outputs:
        assert not (OUT / name).exists(), 'Destination image already exists'
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for case, name in enumerate(outputs, 1):
        paths = []
        for folder in ORDER:
            matches = sorted((SOURCE / folder).glob(f'Case{case}_*.png'))
            assert len(matches) == 1
            p = matches[0]
            assert sha(p) == prior['delivery_hashes'][str(p.relative_to(SOURCE))]
            paths.append(p)
        sizes = []
        for p in paths:
            with Image.open(p) as image:
                sizes.append(image.size)
        assert sizes == [(4296, 2790)] * 5
        canvas = Image.new('RGBA', (4296, 13950), (255, 255, 255, 255))
        for index, p in enumerate(paths):
            with Image.open(p) as image:
                canvas.paste(image.convert('RGBA'), (0, index * 2790))
        output = OUT / name
        canvas.save(output, format='PNG', dpi=(600, 600))
        canvas.close()
        with Image.open(output) as merged:
            assert merged.size == (4296, 13950)
            for index, p in enumerate(paths):
                with Image.open(p) as image:
                    section = merged.crop((0, index * 2790, 4296, (index + 1) * 2790))
                    diff = ImageChops.difference(section, image.convert('RGBA'))
                    assert all(bounds == (0, 0) for bounds in diff.getextrema())
                assert sha(p) == prior['delivery_hashes'][str(p.relative_to(SOURCE))]
        records.append(dict(case=case, file=name, width=4296, height=13950,
                            source_order=[str(p.relative_to(SOURCE)) for p in paths],
                            pixel_identical_sections=True, sha256=sha(output)))
        print('Verified', name, flush=True)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in outputs:
            z.write(OUT / name, arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == outputs and z.testzip() is None
        for r in records:
            assert hashlib.sha256(z.read(r['file'])).hexdigest() == r['sha256']
    report = dict(passed=True, operation='Vertical concatenation without resizing or cropping',
                  models_top_to_bottom=ORDER, outputs=records, source_images_unchanged=True,
                  zip_sha256=sha(archive), code_sha256=sha(Path(__file__)))
    (EVIDENCE / 'vertical_stack_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: 2 case images; all 10 component pixel arrays unchanged; ZIP verified.')


if __name__ == '__main__':
    main()
