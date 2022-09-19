from chiasmus import ChiasmusDetector
import sys


def main():
    fn = sys.argv[1]

    print('initialize detector')
    chidect = ChiasmusDetector(
            fasttext_model = './fasttext_models/wiki.de.bin',
            feature_types = ['dubremetz', 'lexical', 'embedding'],
            conjlist = ["und", "so", "weil", "weder", "noch", "aber", "für", "dennoch"],
            neglist = ["nein", "nicht", "niemals", "nichts"],
            pos_blacklist=["SPACE", "PUNCT", "PROPN", "DET"],
            spacy_model = 'de_core_news_lg'
            )

    print('train with crossvalidation')
    chidect.train_with_crossval(
            training_file='data_example/data.json',
            num_runs=5
            )

    chidect.print_summary()

    print('train on whole dataset')
    chidect.train(
            training_file='data_example/data.json', 
            keep_model=True
            )

    print('find chaismi in new text')
    chidect.run_pipeline_on_text(
            filename=f'{fn}.txt', 
            text_folder="gerdracor",
            processed_folder="processed",
            candidates_folder="candidates",
            id_start="test_"
            )

    chidect.get_top(
            f'candidates/{fn}.txt.pkl', 
            'results.json',
            100)

if __name__ == "__main__":
    main()
