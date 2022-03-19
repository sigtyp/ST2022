surprise-all:
	st2022 --download --datasets=datasets-surprise.json
	st2022 --prepare --datasets=datasets-surprise.json --datapath=data-surprise --runs=10000
	st2022 --split --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --predict --proportion=0.1 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --predict --proportion=0.2 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --predict --proportion=0.3 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --predict --proportion=0.4 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --predict --proportion=0.5 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.1 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.2 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.3 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.4 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.5 --all --datapath=data-surprise --datasets=datasets-surprise.json

evaluate-training:
	st2022 --evaluate --proportion=0.1 --all --datapath=data --datasets=datasets.json
	st2022 --evaluate --proportion=0.2 --all --datapath=data --datasets=datasets.json
	st2022 --evaluate --proportion=0.3 --all --datapath=data --datasets=datasets.json
	st2022 --evaluate --proportion=0.4 --all --datapath=data --datasets=datasets.json
	st2022 --evaluate --proportion=0.5 --all --datapath=data --datasets=datasets.json

evaluate-surprise:
	st2022 --evaluate --proportion=0.1 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.2 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.3 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.4 --all --datapath=data-surprise --datasets=datasets-surprise.json
	st2022 --evaluate --proportion=0.5 --all --datapath=data-surprise --datasets=datasets-surprise.json

