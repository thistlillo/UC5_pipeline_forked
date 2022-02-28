from bs4 import BeautifulSoup
from collections import defaultdict
import os
import yaml



# keys for accessing the dictionaries returned by subsequent methods
k_identifier = 'id'

k_comparison = 'comparison'
k_filename = 'filename'
k_findings = 'findings'
k_impression = 'impression'
k_indication = 'indication'

k_major_mesh = 'major_mesh'
k_n_major_mesh = 'n_' + k_major_mesh
k_minor_mesh = 'minor_mesh'
k_n_minor_mesh = 'n_' + k_minor_mesh
k_auto_term = 'auto_term'
k_n_auto_term = 'n_' + k_auto_term

# for images, return type is a list of dictionary: one dictionary per image
# e.g., r['images'] [ {'image_caption':'...', 'image_filename':'...'}, ... ]
k_images = 'images'
k_n_images = 'n_images'
k_image_caption = 'image_caption'
k_image_filename = 'image_filename'

list_sep = ";"
csv_sep = "\t"

def parse_id(soup):
    keys = ['pmcid', 'iuxrid', 'uid']
    d = defaultdict(None)
    selected_id = None
    for k in keys:
        if soup(k):
            # since: soup(k) returns:
            #        [<pmcid id="3315"></pmcid>]
            # 1) soup(k)[0] takes the first element of the result set: <pmcid id="3315"></pmcid>
            # 2) soup(k)[0].get('id') reads the value of the property 'id': 3315
            v = soup(k)[0].get('id')
            d[k] = v
            selected_id = v
            if k == keys[0] or k == keys[1]:
                # prefer pmcid or uixrid, that are simple integers. uid starts with 'CXR'
                # example: pmcid=3700, uixrid=3700, uid=CXR3700
                # break as soon as you find one of the first two keys
                break
    assert selected_id  # is not None and is not empty, fail otherwise
    return {k_identifier: selected_id}


def parse_medical_texts(soup):
    a = soup.abstract
    ats = a.find_all('abstracttext')
    res = {}
    valid_labels = [k_impression, k_indication, k_findings, k_comparison]
    for at in ats:
        label = at.get('label').lower()
        if label in valid_labels:
            res[label] = at.text
    return res


def parse_mesh_terms(soup):
    mt = soup.mesh
    res = {}
    if mt:
        mt_major = mt.find_all('major')
        mt_minor = mt.find_all('minor')
        if mt_major:
            res[k_major_mesh] = [major.text for major in mt_major if major.text]
        if mt_minor:
            res[k_minor_mesh] = [minor.text for minor in mt_minor if minor.text]
    return res


def parse_automatic_terms(soup):
    mt = soup.mesh
    res = {}
    terms = []
    if mt:
        mt_auto = mt.find_all('automatic')
        if mt_auto:
            terms = [term.text for term in mt_auto if term.text]
    res[k_auto_term] = terms
    return res


def parse_images(soup):
    d = defaultdict(list)
    imgs = soup.find_all('parentimage')
    if len(imgs) == 0:
        d[k_image_caption] = ""
        d[k_image_filename] = []
    
    for img in imgs:
        if img.caption:
            d[k_image_caption].append(img.caption.text)
        if img.url:
            p = img.url.text  # this is an absolute path
            fn = os.path.basename(p).split(".")[0]
            # dataset contains png images, but paths in reports point to (old) jpeg versions
            d[k_image_filename].append(fn + '.png')
        else:
            print('FATAL: NO img.url')
            exit()
        
    return d


def parse_soup(soup):
    parsed = {}
    # all the following methods return disjoint sets of keys.
    #    we can safely use update(.) to merge dictionaries
    # TODO: optimize, may be update() is not the best way (or simply don't care)
    parsed.update(parse_id(soup))
    parsed.update(parse_medical_texts(soup))
    parsed.update(parse_mesh_terms(soup))
    parsed.update(parse_automatic_terms(soup))
    parsed.update(parse_images(soup))

    # for k, v in parsed.items():
    #    log("Parsed soup, printing keys: %s = %s" % (k, v))
    return parsed
