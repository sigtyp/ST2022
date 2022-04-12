import attr
from pathlib import Path

import pylexibank

from cldfbench import CLDFSpec
from pyclts import CLTS

import lingpy
from clldutils.misc import slug
from unicodedata import normalize


@attr.s
class CustomConcept(pylexibank.Concept):
    Chinese_Gloss = attr.ib(default=None)
    Number = attr.ib(default=None)


@attr.s
class CustomLanguage(pylexibank.Language):
    ChineseName = attr.ib(default=None)
    DialectGroup = attr.ib(default=None)
    SubGroup = attr.ib(default=None)


class Dataset(pylexibank.Dataset):
    dir = Path(__file__).parent
    id = "allenbai"
    concept_class = CustomConcept
    language_class = CustomLanguage

    def cmd_download(self, **kw):
        self.raw_dir.write("sources.bib", pylexibank.getEvoBibAsBibtex("Allen2007", **kw))

    def cldf_specs(self):
        return {
            None: pylexibank.Dataset.cldf_specs(self),
            "structure": CLDFSpec(
                module="StructureDataset",
                dir=self.cldf_dir,
                data_fnames={"ParameterTable": "features.csv"},
            ),
        }

    def cmd_makecldf(self, args):
        with self.cldf_writer(args) as writer:
            wl = lingpy.Wordlist(self.raw_dir.joinpath("Bai-Dialect-Survey.tsv").as_posix())
            writer.add_sources()

            # TODO: add concepts with `add_concepts`
            concept_lookup = {}
            for concept in self.conceptlists[0].concepts.values():
                idx = concept.id.split("-")[-1] + "_" + slug(concept.english)
                writer.add_concept(
                    ID=idx,
                    Name=concept.english,
                    Chinese_Gloss=concept.attributes["chinese"],
                    Number=concept.number,
                    Concepticon_ID=concept.concepticon_id,
                    Concepticon_Gloss=concept.concepticon_gloss,
                )
                concept_lookup[concept.english] = idx
            language_lookup = writer.add_languages(lookup_factory="Name")

            for k in pylexibank.progressbar(wl, desc="wl-to-cldf"):
                if wl[k, "value"]:
                    writer.add_lexemes(
                        Language_ID=language_lookup[wl[k, "doculect"]],
                        Parameter_ID=concept_lookup[wl[k, "concept"]],
                        Value=wl[k, "value"],
                        Source="Allen2007",
                    )
            language_table = writer.cldf["LanguageTable"]

            # Remove column for ISO639P3code since there are no ISO codes.
            writer.cldf["LanguageTable"].tableSchema.columns = [
                col
                for col in writer.cldf["LanguageTable"].tableSchema.columns
                if col.name != "ISO639P3code"
            ]

        with self.cldf_writer(args, cldf_spec="structure", clean=False) as writer:
            # We share the language table across both CLDF datasets:
            writer.cldf.add_component(language_table)
            writer.objects["LanguageTable"] = self.languages
            inventories = self.raw_dir.read_csv(
                "inventories.tsv", normalize="NFC", delimiter="\t", dicts=True
            )

            writer.cldf.add_columns(
                "ParameterTable",
                {"name": "CLTS_BIPA", "datatype": "string"},
                {"name": "CLTS_Name", "datatype": "string"},
                {"name": "Lexibank_BIPA", "datatype": "string"},
                {"name": "Prosody", "datatype": "string"},
            )
            writer.cldf.add_columns("ValueTable", {"name": "Context", "datatype": "string"})

            clts = CLTS(args.clts.dir)
            bipa = clts.transcriptionsystem_dict["bipa"]
            td = clts.transcriptiondata_dict["allenbai"]
            pids, visited = set(), set()
            for row in pylexibank.progressbar(inventories, desc="inventories"):
                for s1, s2, p in zip(
                    row["Value"].split(), row["Lexibank"].split(), row["Prosody"].split()
                ):

                    pidx = (
                        "-".join([str(hex(ord(s)))[2:].rjust(4, "0") for s in row["Value"]])
                        + "_"
                        + p
                    )
                    s1 = normalize("NFD", s1)
                    if not s1 in td.grapheme_map:
                        args.log.warn(
                            "missing sound {0} / {1}".format(
                                s1, " ".join([str(hex(ord(x))) for x in s1])
                            )
                        )
                    else:
                        sound = bipa[td.grapheme_map[s1]]
                        sound_name = sound.name if sound.type not in ["unknown", "marker"] else ""
                        if not pidx in visited:
                            visited.add(pidx)
                            writer.objects["ParameterTable"].append(
                                {
                                    "ID": pidx,
                                    "Name": s1,
                                    "Description": sound_name,
                                    "CLTS_BIPA": td.grapheme_map[s1],
                                    "CLTS_Name": sound_name,
                                    "Lexibank_BIPA": s2,
                                    "Prosody": p,
                                }
                            )

                        if row["Language_ID"] + "_" + pidx in pids:
                            continue
                        else:
                            writer.objects["ValueTable"].append(
                                {
                                    "ID": row["Language_ID"] + "_" + pidx,
                                    "Language_ID": row["Language_ID"],
                                    "Parameter_ID": pidx,
                                    "Value": s1,
                                    "Context": p,
                                    "Source": ["Allen2007"],
                                }
                            )

                        pids.add(row["Language_ID"] + "_" + pidx)
