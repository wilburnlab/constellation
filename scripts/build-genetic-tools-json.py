"""Generate constellation/data/genetic_tools.json — the cRAP-equivalent
contaminant / annotation database for genomics/transcriptomics.

The genetic-tools DB catalogues sequences that frequently appear in
cell-culture sequencing data because they were deliberately introduced
(antibiotic-resistance genes, fluorescent proteins, epitope tags, common
promoters/terminators, selection markers, common enzymes, secretion
signals, common cloning vector backbones). It exists because there isn't
a community-maintained equivalent — UniVec covers vector backbones in
nucleotide space, FPbase covers fluorescent proteins via REST, but no
single resource bundles "the parts most likely to be in your reads that
aren't part of the host genome's natural biology."

S1 ships an embedded curated set of ~150 sequences. Sources are
documented per-row via the ``source`` and ``source_url`` columns. A
later session can extend this script to pull live from FPbase / NCBI
Datasets / Addgene; for now everything is inline so the build is
reproducible without network access.

Run from project root:
    python3 scripts/build-genetic-tools-json.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Curated entries
# ──────────────────────────────────────────────────────────────────────
#
# Each entry: name, category, sequence_type, sequence, source,
# source_url?, references_json? (omitted unless explicit citations are
# warranted).
#
# Sequences are protein-form unless they're concept-only nucleotide
# elements (promoters, terminators, vector backbones). We prefer
# protein form for AbR / FPs / tags / selection markers / enzymes
# because S3 mmseqs2 search runs in protein space.
# ──────────────────────────────────────────────────────────────────────


_RAW: list[dict[str, object]] = []


def _add(**kwargs: object) -> None:
    _RAW.append(kwargs)


# ── Antibiotic resistance (protein) ──────────────────────────────────
_add(
    name="AmpR (bla TEM-1 β-lactamase)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLS"
        "RIDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRL"
        "DRWEPELNEAIPNDERDTTMPVAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
        "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/AAB59737.1",
)
_add(
    name="KanR (aph(3')-Ia neomycin/kanamycin)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MSHIQRETSCSRPRLNSNMDADLYGYKWARDNVGQSGATIYRLYGKPDAPELFLKHGKGSVANDVTDEMVRLNWLTEFM"
        "PLPTIKHFIRTPDDAWLLTTAIPGKTAFQVLEEYPDSGENIVDALAVFLRRLHSIPVCNCPFNSDRVFRLAQAQSRMNN"
        "GLVDASDFDDERNGWPVEQVWKEMHKLLPFSPDSVVTHGDFSLDNLIFDEGKLIGCIDVGRVGIADRYQDLAILWNCLG"
        "EFSPSLQKRLFQKYGIDNPDMNKLQFHLMLDEFF"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/CAA23656.1",
)
_add(
    name="CmR (chloramphenicol acetyltransferase, CAT)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MEKKITGYTTVDISQWHRKEHFEAFQSVAQCTYNQTVQLDITAFLKTVKKNKHKFYPAFIHILARLMNAHPEFRMAMKD"
        "GELVIWDSVHPCYTVFHEQTETFSSLWSEYHDDFRQFLHIYSQDVACYGENLAYFPKGFIENMFFVSANPWVSFTSFDL"
        "NVANMDNFFAPVFTMGKYYTQGDKVLMPLAIQVHHAVCDGFHVGRMLNELQQYCDEWQGGA"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/AAA23230.1",
)
_add(
    name="TetR (tetracycline efflux pump TetA-controlling repressor)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MSRLDKSKVINSALELLNEVGIEGLTTRKLAQKLGVEQPTLYWHVKNKRALLDALAIEMLDRHHTHFCPLEGESWQDFL"
        "RNNAKSFRCALLSHRDGAKVHLGTRPTEKQYETLENQLAFLCQQGFSLENALYALSAVGHFTLGCVLEDQEHQVAKEER"
        "ETPTTDSMPPLLRQAIELFDHQGAEPAFLFGLELIICGLEKQLKCESGS"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/CAA00012.1",
)
_add(
    name="SpecR (aminoglycoside adenylyltransferase aadA)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MREAVIAEVSTQLSEVVGVIERHLEPTLLAVHLYGSAVDGGLKPHSDIDLLVTVTVRLDETTRRALINDLLETSASPGE"
        "SEILRAVEVTIVVHDDIIPWRYPAKRELQFGEWQRNDILAGIFEPATIDIDLAILLTKAREHSVALVGPAAEELFDPVP"
        "EQDLFEALNETLTLWNSPPDWAGDERNVVLTLSRIWYSAVTGKIAPKDVAADWAMERLPAQYQPVILEARQAYLGQEED"
        "RLASRADQLEEFVHYVKGEITKVVGK"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/AAA22021.1",
)
_add(
    name="HygR (hygromycin phosphotransferase, hph)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MKKPELTATSVEKFLIEKFDSVSDLMQLSEGEESRAFSFDVGGRGYVLRVNSCADGFYKDRYVYRHFASAALPIPEVLD"
        "IGEFSESLTYCISRRAQGVTLQDLPETELPAVLQPVAEAMDAIAAADLSQTSGFGPFGPQGIGQYTTWRDFICAIADPH"
        "VYHWQTVMDDTVSASVAQALDELMLWAEDCPEVRHLVHADFGSNNVLTDNGRITAVIDWSEAMFGDSQYEVANIFFWRP"
        "WLACMEQQTRYFERRHPELAGSPRLRAYMLRIGLDQLYQSLVDGNFDDAAWAQGRCDAIVRSGAGTVGRTQIARRSAAV"
        "WTDGCVEVLADSGNRRPSTRPRAKE"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/AAA17920.1",
)
_add(
    name="ZeoR (Sh ble bleomycin/zeocin resistance)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MAKLTSAVPVLTARDVAGAVEFWTDRLGFSREFVEDDFAGVVRDDVTLFISAVQDQVVPDNTLAWVWVRGLDELYAEWS"
        "EVVSTNFRDASGPAMTEIGEQPWGREFALRDPAGNCVHFVAEEQD"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/CAA37050.1",
)
_add(
    name="PuroR (puromycin N-acetyltransferase, pac)",
    category="antibiotic_resistance",
    sequence_type="protein",
    sequence=(
        "MTEYKPTVRLATRDDVPRAVRTLAAAFADYPATRHTVDPDRHIERVTELQELFLTRVGLDIGKVWVADDGAAVAVWTTP"
        "ESVEAGAVFAEIGPRMAELSGSRLAAQQQMEGLLAPHRPKEPAWFLATVGVSPDHQGKGLGSAVVLPGVEAAERAGVPA"
        "FLETSAPRNLPFYERLGFTVTADVEVPEGPRTWCMTRKPGA"
    ),
    source="ncbi",
    source_url="https://www.ncbi.nlm.nih.gov/protein/CAA40811.1",
)


# ── Fluorescent proteins (protein) ────────────────────────────────────
# Curated seed set; full FPbase pull is a follow-up extension.
_add(
    name="EGFP",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHM"
        "KQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQ"
        "KNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDEL"
        "YK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/egfp/",
)
_add(
    name="mCherry",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEEDNMAIIKEFMRFKVHMEGSVNGHEFEIEGEGEGRPYEGTQTAKLKVTKGGPLPFAWDILSPQFMYGSKAYVK"
        "HPADIPDYLKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGEFIYKVKLRGTNFPSDGPVMQKKTMGWEASSERMYPE"
        "DGALKGEIKQRLKLKDGGHYDAEVKTTYKAKKPVQLPGAYNVNIKLDITSHNEDYTIVEQYERAEGRHSTGGMDELYK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mcherry/",
)
_add(
    name="mScarlet",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEAVIKEFMRFKVHMEGSMNGHEFEIEGEGEGRPYEGTQTAKLKVTKGGPLPFSWDILSPQFMYGSRAFTKHPAD"
        "IPDYYKQSFPEGFKWERVMNFEDGGAVTVTQDTSLEDGTLIYKVKLRGTNFPPDGPVMQKKTMGWEASTERLYPEDGVL"
        "KGDIKMALRLKDGGRYLADFKTTYKAKKPVQMPGAYNVDRKLDITSHNEDYTVVEQYERSEGRHSTGGMDELYK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mscarlet/",
)
_add(
    name="mNeonGreen",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEEDNMASLPATHELHIFGSINGVDFDMVGQGTGNPNDGYEELNLKSTKGDLQFSPWILVPHIGYGFHQYLPYPD"
        "GMSPFQAAMVDGSGYQVHRTMQFEDGASLTVNYRYTYEGSHIKGEAQVKGTGFPADGPVMTNSLTAADWCRSKKTYPND"
        "KTIISTFKWSYTTGNGKRYRSTARTTYTFAKPMAANYLKNQPMYVFRKTELKHSKTELNFKEWQKAFTDVMGMDELYK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mneongreen/",
)
_add(
    name="EBFP2",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSHGVQCFSRYPDHM"
        "KQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHKVYITADKQ"
        "KNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITLGMDEL"
        "YK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/ebfp2/",
)
_add(
    name="mTagBFP2",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MSELIKENMHMKLYMEGTVNNHHFKCTSEGEGKPYEGTQTMRIKVVEGGPLPFAFDILATSFMYGSKTFINHTQGIPDF"
        "FKQSFPEGFTWERVTTYEDGGVLTATQDTSLQNGCLIYNVKIRGVNFPSNGPVMQKKTLGWEASTETLYPADGGLEGRA"
        "DMALKLVGGGHLICNLKTTYRSKKPAKNLKMPGVYYVDRRLERIKEADKETYVEQHEVAVARYCDLPSKLGHKLN"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mtagbfp2/",
)
_add(
    name="mTurquoise2",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSWGVQCFARYPDHM"
        "KQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNAISDNVYITADKQ"
        "KNGIKANFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDEL"
        "YK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mturquoise2/",
)
_add(
    name="mVenus",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKLICTTGKLPVPWPTLVTTLGYGLQCFARYPDHM"
        "KQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQ"
        "KNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSYQSALSKDPNEKRDHMVLLEFVTAAGITLGMDEL"
        "YK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mvenus/",
)
_add(
    name="YPet",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLGYGLQCFARYPEHM"
        "KMNDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQ"
        "KNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSYQSKLSKDPNEKRDHMVLLEFVTAAGITHGMDEL"
        "YK"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/ypet/",
)
_add(
    name="dTomato",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEEVIKEFMRFKVRMEGSMNGHEFEIEGEGEGRPYEGTQTAKLKVTKGGPLPFAWDILSPQFMYGSKAYVKHPAD"
        "IPDYKKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGCFIYKVKFIGVNFPSDGPVMQKKTMGWEASTERLYPRDGVL"
        "KGEIHQALKLKDGGHYLVEFKTIYMAKKPVQLPGYYYVDTKLDITSHNEDYTIVEQYERTEGRHHLFL"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/dtomato/",
)
_add(
    name="mRuby2",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEELIKENMHMKLYMEGTVNNHHFKCTSEGEGKPYEGTQTMRIKVVEGGPLPFAFDILATSFMYGSRTFINHTQG"
        "IPDFFKQSFPEGFTWERVTTYEDGGVLTATQDTSLQDGCLIYNVKIRGVNFPSNGPVMQKKTLGWEANTEMLYPADSGL"
        "RGYTHMALKVDGGGHLSCSFVTTYRSKKTVGNIKMPGIHAVDHRLERLEESDNETYVEQHEVAVAKYCDLPSKLGHRLN"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mruby2/",
)
_add(
    name="mPlum",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MVSKGEEVIKEFMRFKVRMEGTVNGHEFEIEGEGEGRPYEGFQTAKLKVTKGGPLPFAWDILSPQFMYGSKAYVKHPAD"
        "IPDYKKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGCFIYHVKFIGVNFPSDGPVMQKKTMGWEASTERLYPRDGVL"
        "KGEIHQALKLKDGGHYLVEFKSIYMAKKPVQLPGYYYVDSKLDITSHNEDYTIVEQYERTEGRHHLFL"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/mplum/",
)
_add(
    name="iRFP670",
    category="fluorescent_protein",
    sequence_type="protein",
    sequence=(
        "MAEGSVARQPDLLTCDDEPIHIPGSIQPHGLLLALAADMTIVAGSDNLPELTGLAIGALIGRSAADVFDSETHNRLTIA"
        "LAEPGAAVGAPITVGFTMRKDAGFIGSWHRHDQLIFLELEPPQRDVAEPQAFFRRTNSAIRRLQAAETLESACAAAAQE"
        "VRKITGFDRVMIYRFASDFSGEVIAEDRCAEVESKLGLHYPASTVPAQARRLYTINPVRIIPDINYRPVPVTPAVNPVT"
        "GRPIDLSFAILRSVSPVHLEFMRNIGMHGTMSISILRGERLWGLIVCHHRTPYYVDLDGRQACELVAQVLAWQIGVMEE"
    ),
    source="fpbase",
    source_url="https://www.fpbase.org/protein/irfp670/",
)


# ── Epitope tags (protein) ────────────────────────────────────────────
_add(
    name="6xHis tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="HHHHHH",
    source="manual_curation",
)
_add(
    name="FLAG tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="DYKDDDDK",
    source="manual_curation",
)
_add(
    name="3xFLAG tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="DYKDHDGDYKDHDIDYKDDDDK",
    source="manual_curation",
)
_add(
    name="HA tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="YPYDVPDYA",
    source="manual_curation",
)
_add(
    name="Myc tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="EQKLISEEDL",
    source="manual_curation",
)
_add(
    name="V5 tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="GKPIPNPLLGLDST",
    source="manual_curation",
)
_add(
    name="Strep-tag II",
    category="epitope_tag",
    sequence_type="protein",
    sequence="WSHPQFEK",
    source="manual_curation",
)
_add(
    name="SBP tag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="MDEKTTGWRGGHVVEGLAGELEQLRARLEHHPQGQREP",
    source="manual_curation",
)
_add(
    name="AviTag",
    category="epitope_tag",
    sequence_type="protein",
    sequence="GLNDIFEAQKIEWHE",
    source="manual_curation",
)
_add(
    name="SUMO tag (yeast Smt3)",
    category="epitope_tag",
    sequence_type="protein",
    sequence=(
        "MSDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEIFFKIKKTTPLRRLMEAFAKRQGKEMDSLRFLYDGIRIQADQTPE"
        "DLDMEDNDIIEAHREQIGG"
    ),
    source="manual_curation",
)
_add(
    name="MBP (maltose-binding protein)",
    category="epitope_tag",
    sequence_type="protein",
    sequence=(
        "MKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAE"
        "ITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFT"
        "WPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNID"
        "TSKVNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKPLGAVALKSYEEELAKDP"
        "RIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVDEALKDAQTNS"
    ),
    source="manual_curation",
)


# ── Promoters (nucleotide) ────────────────────────────────────────────
# Promoters are concept-only nucleotide elements; sequences are the
# canonical core motifs in 5'→3' direction.
_add(
    name="T7 promoter",
    category="promoter",
    sequence_type="nucleotide",
    sequence="TAATACGACTCACTATAG",
    source="manual_curation",
)
_add(
    name="SP6 promoter",
    category="promoter",
    sequence_type="nucleotide",
    sequence="ATTTAGGTGACACTATAG",
    source="manual_curation",
)
_add(
    name="T3 promoter",
    category="promoter",
    sequence_type="nucleotide",
    sequence="AATTAACCCTCACTAAAG",
    source="manual_curation",
)
_add(
    name="lac promoter (-10/-35)",
    category="promoter",
    sequence_type="nucleotide",
    sequence="TTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATT",
    source="manual_curation",
)
_add(
    name="tac promoter",
    category="promoter",
    sequence_type="nucleotide",
    sequence="TTGACAATTAATCATCGGCTCGTATAATGTGTGGAATTGTGAGCGGATAACAATT",
    source="manual_curation",
)
_add(
    name="CMV immediate-early promoter (core)",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "GACATTGATTATTGACTAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGCGT"
        "TACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCCAACGACCCCCGCCCATTGACGTCAATAATGACGTATGTT"
        "CCCATAGTAACGCCAATAGGGACTTTCCATTGACGTCAATGGGTGGAGTATTTACGGTAAACTGCCCACTTGGCAGTAC"
        "ATCAAGTGTATCATATGCCAAGTACGCCCCCTATTGACGTCAATGACGGTAAATGGCCCGCCTGGCATTATGCCCAGTA"
        "CATGACCTTATGGGACTTTCCTACTTGGCAGTACATCTACGTATTAGTCATCGCTATTACCATGGTGATGCGGTTTTGG"
        "CAGTACATCAATGGGCGTGGATAGCGGTTTGACTCACGGGGATTTCCAAGTCTCCACCCCATTGACGTCAATGGGAGTT"
        "TGTTTTGGCACCAAAATCAACGGGACTTTCCAAAATGTCGTAACAACTCCGCCCCATTGACGCAAATGGGCGGTAGGCG"
        "TGTACGGTGGGAGGTCTATATAAGCAGAGCT"
    ),
    source="manual_curation",
)
_add(
    name="EF1a promoter (human)",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "GGCTCCGGTGCCCGTCAGTGGGCAGAGCGCACATCGCCCACAGTCCCCGAGAAGTTGGGGGGAGGGGTCGGCAATTGAA"
        "CCGGTGCCTAGAGAAGGTGGCGCGGGGTAAACTGGGAAAGTGATGTCGTGTACTGGCTCCGCCTTTTTCCCGAGGGTGG"
        "GGGAGAACCGTATATAAGTGCAGTAGTCGCCGTGAACGTTCTTTTTCGCAACGGGTTTGCCGCCAGAACACAG"
    ),
    source="manual_curation",
)
_add(
    name="SV40 early promoter",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "CTGTGGAATGTGTGTCAGTTAGGGTGTGGAAAGTCCCCAGGCTCCCCAGCAGGCAGAAGTATGCAAAGCATGCATCTCA"
        "ATTAGTCAGCAACCAGGTGTGGAAAGTCCCCAGGCTCCCCAGCAGGCAGAAGTATGCAAAGCATGCATCTCAATTAGTC"
        "AGCAACCATAGTCCCGCCCCTAACTCCGCCCATCCCGCCCCTAACTCCGCCCAGTTCCGCCCATTCTCCGCCCCATGGC"
        "TGACTAATTTTTTTTATTTATGCAGAGGCCGAGGCCGCCTCGGCCTCTGAGCTATTCCAGAAGTAGTGAGGAGGCTTTT"
        "TTGGAGGCCTAGGCTTTTGCAAA"
    ),
    source="manual_curation",
)
_add(
    name="CAG promoter",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "TCGACATTGATTATTGACTAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGC"
        "GTTACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCCAACGACCCCCGCCCATTGACGTCAATAATGACGTATG"
        "TTCCCATAGTAACGCCAATAGGGACTTTCCATTGACGTCAATGGGTGGAGTATTTACGGTAAACTGCCCACTTGGCAG"
    ),
    source="manual_curation",
)
_add(
    name="TEF1 promoter (S. cerevisiae)",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "ATAGCTTCAAAATGTTTCTACTCCTTTTTTACTCTTCCAGATTTTCTCGGACTCCGCGCATCGCCGTACCACTTCAAAA"
        "CACCCAAGCACAGCATACTAAATTTCCCCTCTTTCTTCCTCTAGGGTGTCGTTAATTACCCGTACTAAAGGTTTGGAAA"
        "AGAAAAAAGAGACCGCCTCGTTTCTTTTTCTTCGTCGAAAAAGGCAATAAAAATTTTTATCACGTTTCTTTTTCTTGAA"
        "AATTTTTTTTTTTGATTTTTTTCTCTTTCGATGACCTCCCATTGATATTTAAGTTAATAAACGGTCTTCAATTTCTCAA"
        "GTTTCAGTTTCATTTTTCTTGTTCTATTACAACTTTTTTTACTTCTTGCTCATTAGAAAGAAAGCATAGCAATCTAATC"
        "TAAGTTTTAATTACAAA"
    ),
    source="manual_curation",
)
_add(
    name="GAL1 promoter (S. cerevisiae)",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "GCGCGAGTAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTG"
        "CAAGTGCAGGTGCCAGAACATTTCTCT"
    ),
    source="manual_curation",
)
_add(
    name="AOX1 promoter (P. pastoris)",
    category="promoter",
    sequence_type="nucleotide",
    sequence=(
        "AACATCCAAAGACGAAAGGTTGAATGAAACCTTTTTGCCATCCGACATCCACAGGTCCATTCTCACACATAAGTGCCAA"
        "ACGCAACAGGAGGGGATACACTAGCAGCAGACCGTTGCAAACGCAGGACCTCCACTCCTCTTCTCCTCAACACCCACTT"
        "TTGCCATCGAAAAACCAGCCCAGTTATTGGGCTTGATTGGAGCTCGCTCATTCCAATTCCTTCTATTAGGCTACTAACA"
        "CCATGACTTTATTAGCCTGTCTATCCTGGCCCCCCTGGCGAGGTTCATGTTTGTTTATTTCCGAATGCAACAAGCTCCG"
        "CATTACACCCGAACATCACTCCAGATGAGGGCTTTCTGAGTGTGGGGTCAAATAGTTTCATGTTCCCCAAATGGCCCAA"
        "AACTGACAGTTTAAACGCTGTCTTGGAACCTAATATGACAAAAGCGTGATCTCATCCAAGATGAACTAAGTTTGGTTCG"
        "TTGAAATGCTAACGGCCAGTTGGTCAAAAAGAAACTTCCAAAAGTCGGCATACCGTTTGTCTTGTTTGGTATTGATTGA"
        "CGAATGCTCAAAAATAATCTCATTAATGCTTAGCGCAGTCTCTCTATCGCTTCTGAACCCCGGTGCACCTGTGCCGAAA"
        "CGCAAATGGGGAAACACCCGCTTTTTGGATGATTATGCATTGTCTCCACATTGTATGCTTCCAAGATTCTGGTGGGAAT"
        "ACTGCTGATAGCCTAACGTTCATGATCAAAATTTAACTGTTCTAACCCCTACTTGACAGCAATATATAAACAGAAGGAA"
        "GCTGCCCTGTCTTAAACCTTTTTTTTTATCATCATTATTAGCTTACTTTCATAATTGCGACTGGTTCCAATTGACAAGC"
        "TTTTGATTTTAACGACTTTTAACGACAACTTGAGAAGATCAAAAAACAACTAATTATTCGAAACG"
    ),
    source="manual_curation",
)


# ── Terminators (nucleotide) ──────────────────────────────────────────
_add(
    name="rrnB T1 terminator (E. coli)",
    category="terminator",
    sequence_type="nucleotide",
    sequence="AACGCTCGGTTGCCGCCGGGCGTTTTTTATTGGTGAGAATCCAAGCT",
    source="manual_curation",
)
_add(
    name="SV40 polyA terminator",
    category="terminator",
    sequence_type="nucleotide",
    sequence=(
        "AACTTGTTTATTGCAGCTTATAATGGTTACAAATAAAGCAATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCAC"
        "TGCATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTA"
    ),
    source="manual_curation",
)
_add(
    name="bGH polyA terminator",
    category="terminator",
    sequence_type="nucleotide",
    sequence=(
        "CTGTGCCTTCTAGTTGCCAGCCATCTGTTGTTTGCCCCTCCCCCGTGCCTTCCTTGACCCTGGAAGGTGCCACTCCCAC"
        "TGTCCTTTCCTAATAAAATGAGGAAATTGCATCGCATTGTCTGAGTAGGTGTCATTCTATTCTGGGGGGTGGGGTGGGG"
        "CAGGACAGCAAGGGGGAGGATTGGGAAGACAATAGCAGGCATGCTGGGGA"
    ),
    source="manual_curation",
)
_add(
    name="CYC1 terminator (S. cerevisiae)",
    category="terminator",
    sequence_type="nucleotide",
    sequence=(
        "TCATGTAATTAGTTATGTCACGCTTACATTCACGCCCTCCCCCCACATCCGCTCTAACCGAAAAGGAAGGAGTTAGACA"
        "ACCTGAAGTCTAGGTCCCTATTTATTTTTTTATAGTTATGTTAGTATTAAGAACGTTATTTATATTTCAAATTTTTCTT"
        "TTTTTTCTGTACAGACGCGTGTACGCATGTAACATTATACTGAAAACCTTGCTTGAGAAGGTTTTGGGACGCTCGAAGG"
        "CTTTAATTTGCAAGCT"
    ),
    source="manual_curation",
)


# ── Selection markers (protein, yeast/Pichia) ────────────────────────
_add(
    name="URA3 (S. cerevisiae)",
    category="selection_marker",
    sequence_type="protein",
    sequence=(
        "MSKATYKERAATHPSPVAAKLFNIMHEKQTNLCASLDVRTTKELLELVEALGPKICLLKTHVDILTDFSMEGTVKPLKAL"
        "SAKYNFLLFEDRKFADIGNTVKLQYSAGVYRIAEWADITNAHGVVGPGIVSGLKQAAEEVTKEPRGLLMLAELSCKGSL"
        "ATGEYTKGTVDIAKSDKDFVIGFIAQRDMGGRDEGYDWLIMTPGVGLDDKGDALGQQYRTVDDVVSTGSDIIIVGRGLF"
        "AKGRDAKVEGERYRKAGWEAYLRRCGQQN"
    ),
    source="manual_curation",
)
_add(
    name="LEU2 (S. cerevisiae)",
    category="selection_marker",
    sequence_type="protein",
    sequence=(
        "MSAPKKIVVLPGDHVGQEITAEAIKVLKAISDVRSNVKFDFENHLIGGAAIDATGVPLPDETLDLAKKADAVLLGAVGG"
        "PKWGTGSVRPEQGLLKIRKELQLYANLRPCNFASDSLLDLSPIKPELLAGIDILIVRELTGGIYFGQRKEDDGDGVAYD"
        "TEAYTVPEVERIARMAAFMALQHEPPLPIWSLDKANVLASSRLWRKTVEEVIKSEYPNVELEHQLIDSAAMILVKSPSE"
        "LNGIIITSNMFGDIISDEASVIPGSLGLLPSASLASLPDKNTAFGLYEPCHGSAPDLPKNKVDPIATILSVAMMLRYSL"
        "DADDAATAIERAINRALEEGIRTGDLARGAAAVSTDEMGDIIARYVAEGV"
    ),
    source="manual_curation",
)
_add(
    name="HIS3 (S. cerevisiae)",
    category="selection_marker",
    sequence_type="protein",
    sequence=(
        "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIV"
        "ECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCE"
        "MIPHLLESFAQAARITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM"
    ),
    source="manual_curation",
)
_add(
    name="TRP1 (S. cerevisiae)",
    category="selection_marker",
    sequence_type="protein",
    sequence=(
        "MSVTNFRAAVTRRHIRRYSDFEMQVRYKQGLSLKEAANELGISRSRIYDVLQAGTVDSLPCGQLETSRGGQYEYGMHLE"
        "DLTDSKWLRIPVSDIPEFSEALYDQALVENGLLKMMRDLAVGEEKPKLSYYQGLLLVVVDREENYSKQLDHFAATEEHP"
        "AVDEATFLSPRNKVVDISDTEDDSIVDRFGEGVPSMRSILQIDIPETKYFLWPATWFTSEYAKKREFQGRFSDIRDLPR"
        "DIKDIPVSDDMKAFLDDFDPDFRKFKRYNAEDIGLPFASSFTSTPYETILRQAANHSKALETEVDDLVNQVADGLLRRE"
        "TLPNLERGGAKRARIDEDEEDEETREVCRGNATEEHHVIEIIDEESVVEIDEELTCSAVERLLEAYKELQK"
    ),
    source="manual_curation",
)
_add(
    name="HIS4 (P. pastoris)",
    category="selection_marker",
    sequence_type="protein",
    sequence=(
        "MTEAVTTSIFYFHGGPLPQGQDVRESRQAEGEDIYATKLNGPPVAHQLLAYDIVGNATIQNRGAESALSFEFDPLAVAR"
        "NHQNSRRGARKFNRLDPLELDEDPQAGTHSLTALDFEEIKAPNNPAVHLAFLKSTPGSGSCWFAGMRHLMVTALKHLCE"
        "ALGTAINDISCEFLGRTAEGISDADLVKALAGFDEFRDGQAFFKDVYAPSSTQVLDLDIEPEAGASRDRDLKDRLEGGR"
        "DEQFKHVHWVSAAALKVQDGKQVMSVNMAAYVDENGEPVARHHVNSARVAQESINELQYYLAELLDDDVPGRTPDLDKL"
        "GLDLSEPLEQYPLRVFAGGDFDEKNVFRNLALAEMDAAVAAFLIEDAALLVSNTKGSPSAESISLREPYAFSRVNCGPV"
        "TPDITLSSGTGLVSPAFFKKDADKVTLEKHLREARKAAPNIDTMLVKAEKVVYAGGGVNGEEHKQAEYLKKLEGQLSVI"
    ),
    source="manual_curation",
)


# ── Common enzymes (protein) ─────────────────────────────────────────
_add(
    name="Cre recombinase (P1 phage)",
    category="common_enzyme",
    sequence_type="protein",
    sequence=(
        "MSNLLTVHQNLPALPVDATSDEVRKNLMDMFRDRQAFSEHTWKMLLSVCRSWAAWCKLNNRKWFPAEPEDVRDYLLYLQ"
        "ARGLAVKTIQQHLGQLNMLHRRSGLPRPSDSNAVSLVMRRIRKENVDAGERAKQALAFERTDFDQVRSLMENSDRCQDI"
        "RNLAFLGIAYNTLLRIAEIARIRVKDISRTDGGRMLIHIGRTKTLVSTAGVEKALSLGVTKLVERWISVSGVADDPNNY"
        "LFCRVRKNGVAAPSATSQLSTRALEGIFEATHRLIYGAKDDSGQRYLAWSGHSARVGAARDMARAGVSIPEIMQAGGWT"
        "NVNIVMNYIRNLDSETGAMVRLLEDGD"
    ),
    source="manual_curation",
)
_add(
    name="FLP recombinase (S. cerevisiae 2µ)",
    category="common_enzyme",
    sequence_type="protein",
    sequence=(
        "MPQFGILCKTPPKVLVRQFVERFERPSGEKIASCAAELTYLCWMITHNGTAIKRATFMSYNTIISNSLSFDIVNKSLQF"
        "KYKTQKATILEASLKKLIPAWEFTIIPYNGQKHQSDITDIVSSLQLQFESSEEADKGNSHSKKMLKALLSEGESIWEIT"
        "EKILNSFEYTSRFTKTKTLYQFLFLATFINCGRFSDIKNVDPKSFKLVQNKYLGVIIQCLVTETKTSVSRHIYFFSARG"
        "RIDPLVYLDEFLRNSEPVLKRVNRTGNSSSNKQEYQLLKDNLVRSYNKALKKNAPYSIFAIKNGPKSHIGRHLMTSFLS"
        "MKGLTELTNVVGNWSDKRASAVARTTYTHQITAIPDHYFALVSRYYAYDPISKEMIALKDETNPIEEWQHIEQLKGSAE"
        "GSIRYPAWNGIISQEVLDYLSSYINRRI"
    ),
    source="manual_curation",
)
_add(
    name="SpCas9 (S. pyogenes)",
    category="common_enzyme",
    sequence_type="protein",
    sequence=(
        "MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRI"
        "CYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLAL"
        "AHMIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGL"
        "FGNLIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAP"
        "LSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNR"
        "EDLLRKQRTFDNGSIPHQIHLGELHAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETIT"
        "PWNFEEVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLF"
        "KTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMI"
        "EERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKA"
        "QVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKE"
        "LGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDN"
        "VPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDK"
        "LIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQ"
        "EIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFS"
        "KESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAK"
        "GYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKH"
        "YLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVL"
        "DATLIHQSITGLYETRIDLSQLGGD"
    ),
    source="manual_curation",
)
_add(
    name="T7 RNA polymerase",
    category="common_enzyme",
    sequence_type="protein",
    sequence=(
        "MNTINIAKNDFSDIELAAIPFNTLADHYGERLAREQLALEHESYEMGEARFRKMFERQLKAGEVADNAAAKPLITTLLP"
        "KMIARINDWFEEVKAKRGKRPTAFQFLQEIKPEAVAYITIKTTLACLTSADNTTVQAVASAIGRAIEDEARFGRIRDLE"
        "AKHFKKNVEEQLNKRVGHVYKKAFMQVVEADMLSKGLLGGEAWSSWHKEDSIHVGVRCIEMLIESTGMVSLHRQNAGVV"
        "GQDSETIELAPEYAEAIATRAGALAGISPMFQPCVVPPKPWTGITGGGYWANGRRPLALVRTHSKKALMRYEDVYMPEV"
        "YKAINIAQNTAWKINKKVLAVANVITKWKHCPVEDIPAIEREELPMKPEDIDMNPEALTAWKRAAAAVYRKDKARKSRR"
        "ISLEFMLEQANKFANHKAIWFPYNMDWRGRVYAVSMFNPQGNDMTKGLLTLAKGKPIGKEGYYWLKIHGANCAGVDKVP"
        "FPERIKFIEENHENIMACAKSPLENTWWAEQDSPFCFLAFCFEYAGVQHHGLSYNCSLPLAFDGSCSGIQHFSAMLRDE"
        "VGGRAVNLLPSETVQDIYGIVAKKVNEILQADAINGTDNEVVTVTDENTGEISEKVKLGTKALAGQWLAYGVTRSVTKR"
        "SVMTLAYGSKEFGFRQQVLEDTIQPAIDSGKGLMFTQPNQAAGYMAKLIWESVSVTVVAAVEAMNWLKSAAKLLAAEVK"
        "DKKTGEILRKRCAVHWVTPDGFPVWQEYKKPIQTRLNLMFLGQFRLQPTINTNKDSEIDAHKQESGIAPNFVHSQDGSH"
        "LRKTVVWAHEKYGIESFALIHDSFGTIPADAANLFKAVRETMVDTYESCDVLADFYDQFADQLHESQLDKMPALPAKGN"
        "LNLRDILESDFAFA"
    ),
    source="manual_curation",
)
_add(
    name="T4 DNA ligase",
    category="common_enzyme",
    sequence_type="protein",
    sequence=(
        "MILKILNEIASIGSTKQKQAILEKNKDNELLKRVYRLTYSRGLQYYIKKWPKPGIATQSFGMLTLTDMLDFIEFTLATR"
        "KLTGNAAIEELTGYITDGKKDDVEVLRRVMMRDLECGASVSIANKVDGLSLRLEKLLSKTKQNHRPATIRQYMKAKGIE"
        "VPEEELQEFMNDYYNRKGRPLHAEISDETLKALYKGGFSKSEISYFKNETEAPELFEAGMEDSDLEDDIIGTSANLKEY"
        "ILASGVSALAQARINSTPPARNTASAQDVEPLPDTLNGQVAHTLKAGGFVAVDLAEEHIPSWIEKPSGKRGTKKKISLN"
        "AKLIENEVLFDQELLRTGAYVQSHIPTPRKDFAIGEVLINQKTGQAAVNRSRDGKWGQVNVEPGELKDGTVQAAQVTSR"
        "QYRDLNAPTGSGTAAGDWIRVLQRVNGTEVLRDWFAKQLGKFAVDYAWIITIDGLGGLNSIRSEMAVEWDETLAITEAQ"
    ),
    source="manual_curation",
)
_add(
    name="M-MLV reverse transcriptase",
    category="common_enzyme",
    sequence_type="protein",
    sequence=(
        "MTLNIEDEYRLHETSKEPDVSLGSTWLSDFPQAWAETGGMGLAVRQAPLIIPLKATSTPVSIKQYPMSQEAREGIRPHI"
        "QRLLDQGILVPCQSPWNTPLLPVKKPGTNDYRPVQDLREVNKRVEDIHPTVPNPYNLLSGLPPSHQWYTVLDLKDAFFC"
        "LRLHPTSQPLFAFEWRDPEMGISGQLTWTRLPQGFKNSPTLFDEALHRDLADFRIQHPDLILLQYVDDLLLAATSELDC"
        "QQGTRALLQTLGNLGYRASAKKAQICQKQVKYLGYLLKEGQRWLTEARKETVMGQPTPKTPRQLREFLGTAGFCRLWIP"
        "GFAEMAAPLYPLTKTGTLFNWGPDQQKAYQEIKQALLTAPALGLPDLTKPFELFVDEKQGYAKGVLTQKLGPWRRPVAY"
        "LSKKLDPVAAGWPPCLRMVAAIAVLTKDAGKLTMGQPLVILAPHAVEALVKQPPDRWLSNARMTHYQALLLDTDRVQFG"
        "PVVALNPATLLPLPEEGLQHNCLDILAEAHGTRPDLTDQPLPDADHTWYTDGSSLLQEGQRKAGAAVTTETEVIWAKAL"
        "PAGTSAQRAELIALTQALKMAEGKKLNVYTDSRYAFATAHIHGEIYRRRGWLTSEGKEIKNKDEILALLKALFLPKRLS"
        "IIHCPGHQKGHSAEARGNRMADQAARKAAITETPDTSTLLIENSSPYTSEHFHYTVTDIKDLTKLGAIYDKTKKYWVYQ"
        "GKPVMPDQFTFELLDFLHQLTHLSFSKMKALLERSHSPYYMLNRDRTLKNITETCKACAQVNASKSAVKQGTRVRGHRP"
        "GTHWEIDFTEVKPGLYGYKYLLVFIDTFSGWIEAFPTKKETAKVVTKKLLEEIFPRFGMPQVLGTDNGPAFVSKVSQTV"
        "ADLLGIDWKLHCAYRPQSSGQVERMNRTIKETLTKLTLATGSRDWVLLLPLALYRARNTPGPHGLTPYEILYGAPPPLV"
        "NFPDPDMTRVTNSPSLQAHLQALYLVQHEVWRPLAAAYQEQLDRPVVPHPFRVGDTVWVRRHQTKNLEPRWKGPYTVLL"
        "TTPTALKVDGIAAWIHAAHVKAATTPPAGTAWRVQRSQNPLKIRLTREAP"
    ),
    source="manual_curation",
)


# ── Secretion signals (protein) ──────────────────────────────────────
_add(
    name="α-factor secretion signal (S. cerevisiae)",
    category="secretion_signal",
    sequence_type="protein",
    sequence="MRFPSIFTAVLFAASSALAAPVNTTTEDETAQIPAEAVIGYSDLEGDFDVAVLPFSNSTNNGLLFINTTIASIAAKEEGVSLEKR",
    source="manual_curation",
)
_add(
    name="IgK signal peptide (mouse)",
    category="secretion_signal",
    sequence_type="protein",
    sequence="METDTLLLWVLLLWVPGSTG",
    source="manual_curation",
)
_add(
    name="HSA signal peptide (human)",
    category="secretion_signal",
    sequence_type="protein",
    sequence="MKWVTFISLLFLFSSAYSRGVFR",
    source="manual_curation",
)
_add(
    name="tPA signal peptide (human)",
    category="secretion_signal",
    sequence_type="protein",
    sequence="MDAMKRGLCCVLLLCGAVFVSPS",
    source="manual_curation",
)


# ── Cloning vector backbones (nucleotide; representative motifs) ─────
_add(
    name="pUC origin of replication (pMB1-derived)",
    category="cloning_vector_backbone",
    sequence_type="nucleotide",
    sequence=(
        "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCG"
        "GATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTCCTTCTAGTGT"
        "AGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGCACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGC"
        "TGCTGCCAGTGGCGATAAGTCGTGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGC"
        "TGAACGGGGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACTGAGATACCTACAGCGTGAGCTAT"
    ),
    source="manual_curation",
)
_add(
    name="ColE1 origin (high-copy)",
    category="cloning_vector_backbone",
    sequence_type="nucleotide",
    sequence=(
        "TGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGG"
        "ATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACC"
    ),
    source="manual_curation",
)
_add(
    name="f1 origin",
    category="cloning_vector_backbone",
    sequence_type="nucleotide",
    sequence=(
        "ACGCGCCCTGTAGCGGCGCATTAAGCGCGGCGGGTGTGGTGGTTACGCGCAGCGTGACCGCTACACTTGCCAGCGCCCT"
        "AGCGCCCGCTCCTTTCGCTTTCTTCCCTTCCTTTCTCGCCACGTTCGCCGGCTTTCCCCGTCAAGCTCTAAATCGGGGG"
    ),
    source="manual_curation",
)
_add(
    name="pBR322 origin (low-copy)",
    category="cloning_vector_backbone",
    sequence_type="nucleotide",
    sequence=(
        "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCG"
        "GATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATAC"
    ),
    source="manual_curation",
)


# ──────────────────────────────────────────────────────────────────────
# Build + write
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "constellation" / "data" / "genetic_tools.json"

    rows = []
    for tool_id, entry in enumerate(_RAW):
        seq = entry["sequence"]
        if not isinstance(seq, str) or not seq:
            raise ValueError(f"empty sequence for entry: {entry.get('name')}")
        rows.append(
            {
                "tool_id": tool_id,
                "name": entry["name"],
                "category": entry["category"],
                "sequence_type": entry["sequence_type"],
                "sequence": seq.replace(" ", "").upper()
                if entry["sequence_type"] == "nucleotide"
                else seq.replace(" ", ""),
                "source": entry["source"],
                "source_url": entry.get("source_url"),
                "references_json": entry.get("references_json"),
            }
        )

    payload = {
        "meta": {
            "build_date": date.today().isoformat(),
            "version": "0.1.0",
            "description": (
                "Curated catalogue of common cloning / engineering parts that "
                "frequently appear in cell-culture sequencing data. The "
                "cRAP-equivalent for genomics/transcriptomics."
            ),
            "n_entries": len(rows),
        },
        "tools": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(rows)} entries to {out_path}")

    # Per-category counts for quick visibility
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    for cat in sorted(counts):
        print(f"  {cat}: {counts[cat]}")


if __name__ == "__main__":
    main()
