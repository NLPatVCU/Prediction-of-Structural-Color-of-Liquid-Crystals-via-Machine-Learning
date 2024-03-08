# Prediction-of-Structural-Color-of-Liquid-Crystals-via-Machine-Learning

This GitHub repository contains the machine learning regression and classification code utilized in a study focused on predicting the reflected wavelength of liquid crystal (LC) mixtures of cholesteryl esters. LC mixtures are known for their ability to generate structural color due to light reflection from the helical structure of self-assembled molecules. It includes implementations of various regression algorithms such as neural network regression and decision tree regression, as well as relevant classification algorithms if applicable.

## Table of Contents

1. [About](#about)
2. [License](#license)
3. [Usage](#usage)
4. [Citation](#citation)

## About

This repository houses the machine learning regression and classification code used in a study focused on predicting the reflected wavelength of liquid crystal (LC) mixtures of cholesteryl esters. LC mixtures are intriguing materials that exhibit structural color due to light reflection from the helical structure of self-assembled molecules. The study aimed to explore the relationship between LC composition and the position of the selective reflection band, which determines the apparent color of the material.

The research involved investigating various machine-learning approaches to predict the reflected wavelength based on the composition of LC formulations. These approaches included neural network regression, decision tree regression, and potentially relevant classification algorithms if applicable. By comparing the predictive performance of these algorithms to a traditional Scheffe cubic model, the study sought to assess the effectiveness of machine learning models in capturing the complex relationship between LC composition and structural color.

Key findings of the study include the identification of decision tree regression as the most effective model for predicting the position of the selective reflection band. The decision tree regression model demonstrated superior accuracy compared to the Scheffe cubic model, particularly for LC formulations not included in the dataset. These results highlight the potential of machine learning techniques in predicting physical properties of LC formulations and offer insights into the development of materials with tailored structural color properties.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details.

## Usage
Please see the header information in each of the python scripts in the code directory. The colours data is located in the data directory. 

## Citation

If you use this code or find it helpful in your research, please cite the following article:

[@article{nguyen2023prediction,
  title={Prediction of the Structural Color of Liquid Crystals via Machine Learning},
  author={Nguyen, Andrew T and Childs, Heather M and Salter, William M and Filippas, Afroditi V and McInnes, Bridget T and Senecal, Kris and Lawton, Timothy J and D’Angelo, Paola A and Zukas, Walter and Alexander, Todd E and others},
  journal={Liquids},
  volume={3},
  number={4},
  pages={440--455},
  year={2023},
  publisher={MDPI}
}]

