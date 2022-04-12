from pathlib import Path
import lingpy as lp

from clldutils.misc import slug
from pylexibank import Dataset as BaseDataset
from pylexibank.util import getEvoBibAsBibtex
from pylexibank import progressbar
from pylexibank import Concept, Language
import attr

#from pyconcepticon import Concepticon

@attr.s
class CustomConcept(Concept):
    Number = attr.ib(default=None)

class Dataset(BaseDataset):
    dir = Path(__file__).parent
    id = 'listsamplesize'
    concept_class = CustomConcept


    def cmd_makecldf(self, args):
    
        concepts = {}
        wl = lp.Wordlist(self.raw_dir.joinpath('IDS.csv').as_posix())

        for concept in self.conceptlists[0].concepts.values():
            idx = '{0}_{1}'.format(concept.number, slug(concept.english))
            args.writer.add_concept(
                    ID=idx,
                    Number=concept.number,
                    Name=concept.english,
                    Concepticon_ID=concept.concepticon_id,
                    Concepticon_Gloss=concept.concepticon_gloss,
                    )
            concepts[concept.attributes['ids_id'].replace('-', '.').strip('0')] = idx

        languages = args.writer.add_languages(
                lookup_factory="Name", id_factory=lambda x: slug(x['Name']))
        
        args.writer.add_sources()
        for idx in wl:
            lexeme = args.writer.add_form(
                    Language_ID=languages[wl[idx, 'language']],
                    Parameter_ID=concepts[wl[idx, 'ids_id']],
                    Value=wl[idx, 'ortho'],
                    Form=wl[idx, 'ipa'].replace('#', '-'),
                    Source='List2014c',
                    Loan=True if wl[idx, 'cogid'] < 0 else False
                    )
            args.writer.add_cognate(
                    lexeme=lexeme,
                    Cognateset_ID=wl[idx, 'cogid'],
                    Cognate_Detection_Method='expert',
                    Source=['List2014c']
                    )        
