"""Build BibTeX from metadata responses retained beside this script."""
from pathlib import Path
import json,re,html,unicodedata
ROOT=Path(__file__).resolve().parents[2]
SRC=Path(__file__).resolve().parent
records=[]

def tex(s):
    # Keep the archive's original metadata intact; use portable TeX accent macros.
    special={'ı':r'{\i}','Ł':r'{\L}','ł':r'{\l}','ß':r'{\ss}','ø':r'{\o}','Ø':r'{\O}','æ':r'{\ae}','Æ':r'{\AE}','œ':r'{\oe}','Œ':r'{\OE}','–':'--','—':'---','’':"'",'−':'-','×':r'$\times$'}
    accent={'\u0300':'`','\u0301':"'",'\u0302':'^','\u0303':'~','\u0308':'"','\u030a':'r','\u030c':'v','\u0306':'u','\u0304':'=','\u0327':'c','\u0328':'k','\u0307':'.'}
    s=html.unescape(str(s));s=s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
    for c,r in special.items():s=s.replace(c,r)
    out='';cs=list(s);i=0
    while i<len(cs):
        c=cs[i];n=unicodedata.normalize('NFD',c)
        if len(n)==2 and n[1] in accent:out+='\\'+accent[n[1]]+'{'+n[0]+'}'
        else:out+=c
        i+=1
    return out

def entry(typ,key,fields,source):
    records.append({'key':key,'type':typ,'source':source,'doi':fields.get('doi')})
    return '@'+typ+'{'+key+',\n'+''.join('  '+k+' = {'+v+'},\n' for k,v in fields.items())+'}\n'

def epmc(key,filename,doi):
    data=json.loads((SRC/filename).read_text())['resultList']['result']
    found=[x for x in data if x.get('doi','').lower()==doi.lower()]
    assert len(found)==1,(key,doi)
    x=found[0];authors=[]
    for a in x.get('authorList',{}).get('author',[]):
        if a.get('lastName'):authors.append(tex(a['lastName'])+', '+tex(a.get('firstName',a.get('initials',''))))
        elif a.get('fullName'):authors.append('{'+tex(a['fullName'])+'}')
    ji=x['journalInfo'];f={'author':' and '.join(authors),'title':'{'+tex(x['title'].rstrip('.'))+'}','journal':tex(ji['journal']['title']),'year':str(x['pubYear'])}
    for a,b in [('volume','volume'),('issue','number')]:
        if ji.get(a):f[b]=tex(ji[a])
    if x.get('pageInfo'):f['pages']=str(x['pageInfo']).replace('-','--')
    f['doi']=doi;f['url']='https://doi.org/'+doi
    return entry('article',key,f,filename+' [Europe PMC core metadata]')

parts=['% Verified bibliography for the Medium976 seed44 MIA draft.\n% Metadata source responses are preserved in sources/official/.\n% Last live verification: 2026-11-02.\n']
items=[
('ctneobc2014','europepmc_cortazar.json','10.1016/S0140-6736(13)62422-8'),
('hylton2012mri','europepmc_hylton2012.json','10.1148/radiol.12110748'),
('hylton2016ftv','europepmc_hylton2016.json','10.1148/radiol.2015150013'),
('mamamia2025','europepmc_mamamia2025.json','10.1038/s41597-025-04707-4'),
('ACRIN6698diffusion2018','europepmc_ACRIN6698diffusion2018.json','10.1148/radiol.2018180273'),
('li2020ispy2','europepmc_ispy2.json','10.1038/s41523-020-00203-7'),
('han2024synthesis','europepmc_seq2seq.json','10.1016/j.media.2023.103044'),
('du2025early','europepmc_longitudinalmri.json','10.1186/s13058-025-02009-6'),
('tang2025spatiotemporal','europepmc_longitudinalmri.json','10.1016/j.eclinm.2025.103298'),
('peng2022pretreatment','europepmc_breastdl2020.json','10.3389/fonc.2022.846775'),
('khan2022review','europepmc_breastdl2020.json','10.3390/tomography8060232'),
]
for x in items:parts.append(epmc(*x))
for key,fn,doi in [('ispy2trial2016','europepmc_ispy2trial2016.json','10.1056/NEJMoa1513750'),('rugo2016ispy2','europepmc_rugo2016.json','10.1056/NEJMoa1513749')]:
    if (SRC/fn).exists() and json.loads((SRC/fn).read_text()).get('hitCount',0):parts.append(epmc(key,fn,doi))
parts.append(entry('misc','tciaispy2_2022',{
 'author':'Li, W. and Newitt, D. C. and Gibbs, J. and Wilmes, L. J. and Jones, E. F. and Arasu, V. A. and others',
 'title':'{I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2), Version 1}',
 'year':'2022','publisher':'The Cancer Imaging Archive','doi':'10.7937/TCIA.D8Z0-9T85','url':'https://doi.org/10.7937/TCIA.D8Z0-9T85','note':'Data set'},'tcia_ispy2.html [TCIA collection citation]'))
parts.append(entry('misc','tciaacrin6698_2021',{
 'author':'Newitt, D. C. and Partridge, S. C. and Zhang, Z. and Gibbs, J. and Chenevert, T. and Rosen, M. and others',
 'title':'{ACRIN 6698/I-SPY2 Breast DWI}',
 'year':'2021','publisher':'The Cancer Imaging Archive','doi':'10.7937/TCIA.KK02-6D95','url':'https://doi.org/10.7937/TCIA.KK02-6D95','note':'Data set'},'tcia_acrin6698.html [TCIA collection citation]'))
x=json.loads((SRC/'crossref_query_jepa2023.json').read_text())['message']['items'][0]
parts.append(entry('inproceedings','jepa2023',{
 'author':' and '.join(tex(a['family'])+', '+tex(a.get('given','')) for a in x['author']),
 'title':'{'+tex(x['title'][0])+'}','booktitle':tex(x['container-title'][0]),'year':'2023','pages':str(x['page']).replace('-','--'),'doi':x['DOI'],'url':'https://doi.org/'+x['DOI']},'crossref_query_jepa2023.json [exact title match]'))
parts.append(entry('inproceedings','vqgan2021',{
 'author':'Esser, Patrick and Rombach, Robin and Ommer, Bjorn',
 'title':'{Taming Transformers for High-Resolution Image Synthesis}',
 'booktitle':'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition','year':'2021','pages':'12873--12883',
 'url':'https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html'},'cvf_vqgan.html [official CVF BibTeX]'))
for key,fn,year,venue in [('vaswani2017','arxiv_transformer.html','2017','Advances in Neural Information Processing Systems'),('adamw2019','arxiv_adamw.html','2019','International Conference on Learning Representations')]:
    s=(SRC/fn).read_text();tags={}
    for k,v in re.findall(r'<meta name="(citation_[^"]+)" content="([^"]*)"',s):tags.setdefault(k,[]).append(html.unescape(v))
    f={'author':' and '.join(tex(a) for a in tags['citation_author']),'title':'{'+tex(tags['citation_title'][0])+'}','booktitle':venue,'year':year,'url':'https://arxiv.org/abs/'+tags['citation_arxiv_id'][0]}
    parts.append(entry('inproceedings',key,f,fn+' [official arXiv author/title metadata; conference venue]'))
(ROOT/'references.bib').write_text('\n'.join(parts))
(SRC/'bibliography_verification.json').write_text(json.dumps({'verified_at':'2026-11-02','entries':records},ensure_ascii=False,indent=2)+'\n')
print('entries',len(records));print('keys',','.join(x['key'] for x in records))
