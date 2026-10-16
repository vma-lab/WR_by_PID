# Datasets

All datasets are public; none are redistributed here. Each is subject to
the terms of its own release. Prepare them under data/ with the layouts
below. Every folder except data/histwi_train_bin,
data/bullinger_teacher_bin and data/BullingerDB holds one sub-folder per writer: the
evaluation takes the sub-folder name as the writer label, and the Stage-2
trainer and the in-training monitoring read the same layout.

Writer folder names are free, but keep the data/ folder names and the
original file names: the per-query
results are keyed by file stem, so a fresh run pairs with the CSVs in
results/ only if the stems are unchanged.

Where a dataset names its files `<writer><separator><rest>`, the writer
folders can be built with the following shell function (no dedicated
builder is shipped for these layouts):

    folderise () {  # source dir, target dir, separator
      mkdir -p "$2"
      for f in "$1"/*; do
        [ -f "$f" ] || continue
        b=$(basename "$f"); w=${b%%"$3"*}
        mkdir -p "$2/$w"
        [ -e "$2/$w/$b" ] || cp "$f" "$2/$w/"
      done
    }

## HisIR19 (ICDAR 2019 Competition on Image Retrieval for Historical Handwritten Documents)

Source: https://zenodo.org/record/3262372 (Christlein et al., ICDAR 2019,
doi:10.1109/ICDAR.2019.00242).

    data/wi_comp_19_validation/<writer>/<page>   1,200 pages, 520 writers
                                                 (the released training set;
                                                 used here only to fit the
                                                 codebook, PCA and exemplar-SVM
                                                 negatives, and for monitoring)
    data/wi_comp_19_test/<writer>/<page>.jpg     20,000 pages. 12,499 pages have at
                                                 least one other page by their
                                                 writer and are scored as queries
                                                 

The pages are named by number (e.g. 17528.jpg); the writer comes from the
competition's ground-truth CSV of each set. Build each folder with

    python tools/reorganize_by_writer.py --input_dir <extracted test pages> \
        --csv <test ground-truth CSV> --output_dir data/wi_comp_19_test

and likewise for the validation set (the CSV needs a file-name column and
a writer column; see the tool's help).

## Historical-WI (ICDAR 2017 Competition on Historical Document Writer Identification)

Source: doi:10.5281/zenodo.1324999 (Fiel et al., ICDAR 2017,
doi:10.1109/ICDAR.2017.225), colour and provided binarised pages of the
training and test sets.

    data/wi_icdar_17_train_colour/<writer>/   1,182 training pages, 394 writers
    data/wi_icdar_17_test_colour/<writer>/    3,600 test pages, 720 writers
    data/wi_icdar_17_train_bin/<writer>/      the provided binarised training
                                              pages as published (white ink on
                                              black), 1,182 pages, 394 folders
    data/wi_icdar_17_test_bin/<writer>/       the provided binarised test pages
                                              as published, 3,600 pages, 720
                                              folders
    data/wi_icdar_17_train_bin_inv/           the two binarised sets inverted to
    data/wi_icdar_17_test_bin_inv/            black ink on white (PNG, same
                                              folders and stems), the polarity
                                              the evaluation expects
    data/histwi_train_bin/                    the provided binarised training
                                              pages in one flat directory, as
                                              the reference trainer expects
                                              (Stage 1)

File names start with the writer number followed by '-' (test pages e.g.
31-IMG_MAX_1005688.jpg, training pages e.g. 3188-3-IMG_MAX_975472.jpg), so
the four writer-folder sets are built with `folderise <extracted pages>
data/wi_icdar_17_test_colour -` and likewise for the other three. Then

    python tools/invert_histwi_bins.py      # builds the two *_bin_inv sets
                                            # and verifies counts and polarity
    mkdir -p data/histwi_train_bin
    find data/wi_icdar_17_train_bin -type f \( -iname '*.png' -o -iname '*.jpg' \
        -o -iname '*.tif' \) -exec cp {} data/histwi_train_bin/ \;
    ls data/histwi_train_bin | wc -l        # expect 1182

data/histwi_train_bin keeps the published polarity (white ink on black)
and the original file names; the reference trainer reads the writer from
the file-name prefix and accepts no other files in the directory.

## CVL (CVL-Database, cropped handwriting fields)

Source: https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/
(Kleber et al., ICDAR 2013).

    data/cvl_eval/<writer>/<writer>-<text>-cropped.tif
                     the cropped pages of the official test set: 1,415 pages,
                     283 writers with 5 pages each

Build with `folderise <test-set cropped pages> data/cvl_eval -`. All
fitting uses the HisIR19 validation set, so the CVL training set is not
needed: reproduce/table6_cvl.sh fits the codebook, PCA and exemplar-SVM
negatives on data/wi_comp_19_validation with the foreground gate disabled
(INK_THRESHOLD=0 applies to the whole run, fits included).

## GRK-Papyri

Source: https://d-scribes.philhist.unibas.ch/en/gkr-papyri/ (Mohammed et
al., ICDAR 2019, doi:10.1109/ICDAR.2019.00121).

    data/grk_papyri/<scribe>/<scribe>_<n>.<ext>
                     50 photographs, 10 scribes (four to seven each),
                     e.g. Andreas/Andreas_8.jpg
    data/grk_papyri_mask/<scribe>/<scribe>_<n>.png
                     one binary mask per photograph, used for token
                     selection only, never as model input

Build the photographs with `folderise <extracted images> data/grk_papyri _`.

The masks are built from the binarised GRK-120 pages of the KaiRacters
release (D-Scribes project, University of Basel,
https://d-scribes.philhist.unibas.ch/en/case-studies/dioscorus/kairacters/;
Peer et al., ICPR 2024, doi:10.1007/978-3-031-78495-8_5). GRK-120 contains
the 50 photographs under the same file stems and pixel sizes.
For each of the 50 photographs take the published binarised page with the
same file stem, threshold it at grey level 128, take the minority class as
foreground, and save it unscaled in the photograph's pixel frame as a
strict 0/255 PNG with foreground 255 (white on black), under the same
writer folder and stem as the photograph. No builder is shipped for this
step.

The evaluation (INK_MASK_DIR=data/grk_papyri_mask) finds each mask by file
stem, tiles it on the same grid as its photograph without any further
binarisation, and takes the minority class of each mask as foreground, so
either polarity is read the same way. A patch token is kept when at least
4% of its 16x16 patch is foreground in the mask (INK_THRESHOLD=0.04). A
missing mask or a tile-grid mismatch stops the run.

## BullingerDB (writer-retrieval split)

Source: Peer et al., ICDAR 2026, doi:10.1007/978-3-032-36033-5_33; data
release: https://zenodo.org/records/19728926. The release provides the colour pages
(bullingerdb-pages, organised by letter), the binarised writer-retrieval
split (bullingerdb-wr, with flat train/ and test/ directories of files
named WRITER_LETTER_PAGE.png) and letters.csv (with the columns
letter_id, writer_id and year).

    data/BullingerDB/                     the extracted release:
                                          bullingerdb-pages/, bullingerdb-wr/,
                                          letters.csv
    data/bullinger_wr_train_colour/p<WRITER>/<LETTER>_<PAGE>.<ext>
                                          the colour page of every training
                                          file of the split: 4,396 pages,
                                          164 writers
    data/bullinger_wr_test_colour/p<WRITER>/<LETTER>_<PAGE>.<ext>
                                          the same for the test split:
                                          16,502 pages, 632 writers
    data/bullinger_wr_train_colour05/     the two colour sets at 0.5 scale
    data/bullinger_wr_test_colour05/      (JPEG, same folders and stems)
    data/bullinger_teacher_bin/           the provided binarised training
                                          pages at 0.5 scale, white ink on
                                          black, one flat directory, named
                                          WRITER-LETTER_PAGE.png
    data/bullinger_letters.csv            a copy of letters.csv

No builder is shipped for the two colour sets: for every file
WRITER_LETTER_PAGE.png of bullingerdb-wr/**/train/ (and likewise test/),
copy the colour page PAGE of letter LETTER from bullingerdb-pages to
data/bullinger_wr_train_colour/p<WRITER>/<LETTER>_<PAGE>.<ext>, keeping
its extension. Then

    python tools/prepare_bullinger_scale05.py      # builds the two *_colour05 sets
    python tools/prepare_bullinger_teacher_bin.py  # builds bullinger_teacher_bin
                                                   # and data/bullinger_letters.csv

Both tools verify the file counts. tools/bullinger_ndcg.py reads the
letter of each query from its file name (<LETTER>_<PAGE>) and the year from
data/bullinger_letters.csv. The Stage-2 student pairs each colour page
with its binarised teacher view through the WRITER, LETTER and PAGE parts
of the two names.
