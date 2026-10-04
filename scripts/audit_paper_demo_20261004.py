"""Read-only artifact/evidence checks, with one internal audit report output."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import zipfile
from PIL import Image,PdfParser
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/images/paper_demo_20261004'
EVIDENCE=ROOT/'docs/internal/2026-10-04_paper_demo_evidence'


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def resolve(pdf,value):
    return pdf.read_indirect(value) if isinstance(value,PdfParser.IndirectReference) else value


def verify_pdf(path,n,width=None,height=None):
    with PdfParser.PdfParser(filename=str(path),mode='rb') as pdf:
        assert len(pdf.pages)==n
        for ref in pdf.pages:
            page=pdf.read_indirect(ref)
            if width is not None:np.testing.assert_allclose(page[b'MediaBox'],[0,0,width*72,height*72],atol=1e-6)
            resources=resolve(pdf,page[b'Resources']);fonts=resolve(pdf,resources[b'Font']);assert fonts
            for ref in fonts.values():
                font=resolve(pdf,ref);assert font[b'Subtype']==PdfParser.PdfName(b'Type0')
                for child in font[b'DescendantFonts']:
                    desc=resolve(pdf,resolve(pdf,child)[b'FontDescriptor']);assert b'FontFile2' in desc
    return True


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='img':self.images.append(d['src'])
        for key in ('href','src'):
            if key in d:self.targets.append(d[key])


def main():
    validation=read(EVIDENCE/'validation.json');rows=read(OUT/'figure_metadata.json')
    assert validation['passed'] and len(rows)==23
    for name,digest in validation['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,digest in validation['artifact_hashes'].items():assert sha(OUT/name)==digest,name
    checks={r['figure']:r for r in validation['render_checks']}
    for r in rows:
        stem=r['stem'];c=checks[stem];assert c['minimum_font_pt']>=8.5 and c['all_text_within_canvas']
        verify_pdf(OUT/f'{stem}.pdf',1,c['width_inches'],c['height_inches'])
        with Image.open(OUT/f'{stem}.png') as image:
            assert image.size==(round(c['width_inches']*600),round(c['height_inches']*600));image.verify()
        svg=(OUT/f'{stem}.svg').read_text()
        assert '<text' in svg and '<path' in svg and '<image' in svg
        assert 'J1' not in svg.split('<image')[0]  # captions/labels independently checked by the renderer
    verify_pdf(OUT/'paper_figures.pdf',23)
    # Compare all numeric main-panel results with their original evaluation rows.
    folders={'own':'flash_external_20261004_r1','usdrl_ntu60':'usdrl_external_20261004_r1',
             'hfd_reproduction':'hfd_reproduction_20261004_r1','privacy_x3d_uda_rgb':'privacy_x3d_external_20261004_r1',
             'flash':'flash_external_20261004_r1'}
    source={m:{r['id']:r for r in read(ROOT/'data/fall_processed/RGB'/folder/'evaluation.json')['rows'] if r['model']==m}
            for m,folder in folders.items()}
    for figure in rows:
        for panel in figure.get('panels',[]):
            assert len(panel['models'])==5 and len(panel['frames'])==4
            for m in panel['models']:
                row=source[m['model_key']][panel['id']]
                assert m['processed']==row['processed'] and m['video_prediction']==row['video_prediction']
                if 'event' in row:
                    assert m['event']=={k:row['event'][k] for k in ('tp','fp','fn')}
                    assert m['alarm_times']==[p['time'] for p in row['predictions']]
                else:assert m['alarm_times'] is None and m['positive_output_times'] is None
    sup=[r for r in rows if r['kind']=='supplementary'];assert len(sup)==20
    frozen=read(EVIDENCE/'selection.json')['supplementary_cases']
    for r,s in zip(sup,frozen):
        assert (r['id'],r['model_key'],r['case'])==(s['id'],s['model'],s['case'])
        assert r['video_prediction']==source[r['model_key']][r['id']]['video_prediction']
    parser=Links();parser.feed((OUT/'index.html').read_text());assert len(parser.images)==23
    for name in parser.targets:assert (OUT/name).is_file(),name
    with zipfile.ZipFile(OUT/'paper_demo_figures.zip') as z:
        assert len(z.namelist())==74 and z.testzip() is None
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==validation['artifact_hashes'][name]
    legacy=ROOT/'docs/internal/2026-10-04_fall_demo_cases_evidence/validation.json'
    for name,digest in read(legacy)['artifact_hashes'].items():assert sha(ROOT/'docs/images/fall_demo_20261004'/name)==digest,name
    result=dict(passed=True,main_figures=3,supplementary_cases=20,pdf_pages=23,
                all_24_pdf_files_have_embedded_vector_fonts=True,all_23_pngs_are_600dpi_at_intended_size=True,
                all_23_svgs_have_text_paths_and_images=True,minimum_font_pt=8.5,
                main_comparison_outcomes_match_original=True,supplementary_selection_unchanged=True,
                legacy_demo_artifacts_unchanged=True,source_files_verified=len(validation['source_hashes']),
                archive_entries_verified=74,gallery_images_verified=23,
                audit_code_sha256=sha(Path(__file__)),validation_sha256=sha(EVIDENCE/'validation.json'))
    (EVIDENCE/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
