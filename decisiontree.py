{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPJtSJdVjWrflXfLcGkPOIS",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/nabiltoby/DataMining/blob/main/decisiontree.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "Wd4TUWdEauki"
      },
      "outputs": [],
      "source": [
        "#Import library numpy, pandas dan sckit-learn\n",
        "import numpy as  np\n",
        "import pandas as pd\n",
        "from sklearn import tree"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Membaca Dataset dari File ke Pandas dataFrame\n",
        "irisDataset = pd.read_csv('Dataset Iris.csv', delimiter=';', header=0)"
      ],
      "metadata": {
        "id": "7uiXaSnWbQh6"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "irisDataset.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "bB0YXwOTdSZP",
        "outputId": "5e886f1a-bb49-4ed9-c6e1-e338fb3f7880"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  \\\n",
              "0   1           7.00          3.02           4.07          1.04   \n",
              "1   2           6.04          3.02           4.05          1.05   \n",
              "2   3           6.09          3.01           4.09          1.05   \n",
              "3   4           5.05          2.03           4.00          1.03   \n",
              "4   5           6.05          2.08           4.06          1.05   \n",
              "\n",
              "           Species  \n",
              "0  Iris-versicolor  \n",
              "1  Iris-versicolor  \n",
              "2  Iris-versicolor  \n",
              "3  Iris-versicolor  \n",
              "4  Iris-versicolor  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f6ba96cd-043f-4f10-afbe-0df4ffa22c75\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Id</th>\n",
              "      <th>SepalLengthCm</th>\n",
              "      <th>SepalWidthCm</th>\n",
              "      <th>PetalLengthCm</th>\n",
              "      <th>PetalWidthCm</th>\n",
              "      <th>Species</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>7.00</td>\n",
              "      <td>3.02</td>\n",
              "      <td>4.07</td>\n",
              "      <td>1.04</td>\n",
              "      <td>Iris-versicolor</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>6.04</td>\n",
              "      <td>3.02</td>\n",
              "      <td>4.05</td>\n",
              "      <td>1.05</td>\n",
              "      <td>Iris-versicolor</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>6.09</td>\n",
              "      <td>3.01</td>\n",
              "      <td>4.09</td>\n",
              "      <td>1.05</td>\n",
              "      <td>Iris-versicolor</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>5.05</td>\n",
              "      <td>2.03</td>\n",
              "      <td>4.00</td>\n",
              "      <td>1.03</td>\n",
              "      <td>Iris-versicolor</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>6.05</td>\n",
              "      <td>2.08</td>\n",
              "      <td>4.06</td>\n",
              "      <td>1.05</td>\n",
              "      <td>Iris-versicolor</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f6ba96cd-043f-4f10-afbe-0df4ffa22c75')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f6ba96cd-043f-4f10-afbe-0df4ffa22c75 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f6ba96cd-043f-4f10-afbe-0df4ffa22c75');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Mengubah kelas (kolom \"Species\") = dari String ke Unique-Integer\n",
        "irisDataset[\"Species\"] = pd.factorize(irisDataset.Species)[0]"
      ],
      "metadata": {
        "id": "_HHPnj8odVt-"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "irisDataset.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "7Xe_aEqSd0R4",
        "outputId": "89cf414e-eff4-4053-ef0d-74122b085a77"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Species\n",
              "0   1           7.00          3.02           4.07          1.04        0\n",
              "1   2           6.04          3.02           4.05          1.05        0\n",
              "2   3           6.09          3.01           4.09          1.05        0\n",
              "3   4           5.05          2.03           4.00          1.03        0\n",
              "4   5           6.05          2.08           4.06          1.05        0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ba5ffe18-6c13-4613-9984-0ddd32ead4a5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Id</th>\n",
              "      <th>SepalLengthCm</th>\n",
              "      <th>SepalWidthCm</th>\n",
              "      <th>PetalLengthCm</th>\n",
              "      <th>PetalWidthCm</th>\n",
              "      <th>Species</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>7.00</td>\n",
              "      <td>3.02</td>\n",
              "      <td>4.07</td>\n",
              "      <td>1.04</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>6.04</td>\n",
              "      <td>3.02</td>\n",
              "      <td>4.05</td>\n",
              "      <td>1.05</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>6.09</td>\n",
              "      <td>3.01</td>\n",
              "      <td>4.09</td>\n",
              "      <td>1.05</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>5.05</td>\n",
              "      <td>2.03</td>\n",
              "      <td>4.00</td>\n",
              "      <td>1.03</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>6.05</td>\n",
              "      <td>2.08</td>\n",
              "      <td>4.06</td>\n",
              "      <td>1.05</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ba5ffe18-6c13-4613-9984-0ddd32ead4a5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ba5ffe18-6c13-4613-9984-0ddd32ead4a5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ba5ffe18-6c13-4613-9984-0ddd32ead4a5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print (irisDataset)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z8yIqaoVeH3Z",
        "outputId": "6733a340-7443-4042-f559-5490b80ead48"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Species\n",
            "0     1           7.00          3.02           4.07          1.04        0\n",
            "1     2           6.04          3.02           4.05          1.05        0\n",
            "2     3           6.09          3.01           4.09          1.05        0\n",
            "3     4           5.05          2.03           4.00          1.03        0\n",
            "4     5           6.05          2.08           4.06          1.05        0\n",
            "..  ...            ...           ...            ...           ...      ...\n",
            "95   96           6.07          3.00           5.02          2.03        1\n",
            "96   97           6.03          2.05           5.00          1.09        1\n",
            "97   98           6.05          3.00           5.02          2.00        1\n",
            "98   99           6.02          3.04           5.04          2.03        1\n",
            "99  100           5.09          3.00           5.01          1.08        1\n",
            "\n",
            "[100 rows x 6 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#meghapus kolom \"Id\"\n",
        "irisDataset = irisDataset.drop(labels=\"Id\", axis=1)"
      ],
      "metadata": {
        "id": "rR69Y0jgd4HJ"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(irisDataset)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ju42HOdYehqu",
        "outputId": "0b93e9c9-70ba-4f63-e0fa-97a5d28a717a"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Species\n",
            "0            7.00          3.02           4.07          1.04        0\n",
            "1            6.04          3.02           4.05          1.05        0\n",
            "2            6.09          3.01           4.09          1.05        0\n",
            "3            5.05          2.03           4.00          1.03        0\n",
            "4            6.05          2.08           4.06          1.05        0\n",
            "..            ...           ...            ...           ...      ...\n",
            "95           6.07          3.00           5.02          2.03        1\n",
            "96           6.03          2.05           5.00          1.09        1\n",
            "97           6.05          3.00           5.02          2.00        1\n",
            "98           6.02          3.04           5.04          2.03        1\n",
            "99           5.09          3.00           5.01          1.08        1\n",
            "\n",
            "[100 rows x 5 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Mengubah dataFrame ke array Numpy\n",
        "irisDataset = irisDataset.to_numpy()"
      ],
      "metadata": {
        "id": "Cy9qnmjDef7r"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(irisDataset)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6esQVH8Je1R7",
        "outputId": "59c32220-2ae4-4b52-f4b5-2dd4c4fd9d77"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[7.   3.02 4.07 1.04 0.  ]\n",
            " [6.04 3.02 4.05 1.05 0.  ]\n",
            " [6.09 3.01 4.09 1.05 0.  ]\n",
            " [5.05 2.03 4.   1.03 0.  ]\n",
            " [6.05 2.08 4.06 1.05 0.  ]\n",
            " [5.07 2.08 4.05 1.03 0.  ]\n",
            " [6.03 3.03 4.07 1.06 0.  ]\n",
            " [4.09 2.04 3.03 1.   0.  ]\n",
            " [6.06 2.09 4.06 1.03 0.  ]\n",
            " [5.02 2.07 3.09 1.04 0.  ]\n",
            " [5.   2.   3.05 1.   0.  ]\n",
            " [5.09 3.   4.02 1.05 0.  ]\n",
            " [6.   2.02 4.   1.   0.  ]\n",
            " [6.01 2.09 4.07 1.04 0.  ]\n",
            " [5.06 2.09 3.06 1.03 0.  ]\n",
            " [6.07 3.01 4.04 1.04 0.  ]\n",
            " [5.06 3.   4.05 1.05 0.  ]\n",
            " [5.08 2.07 4.01 1.   0.  ]\n",
            " [6.02 2.02 4.05 1.05 0.  ]\n",
            " [5.06 2.05 3.09 1.01 0.  ]\n",
            " [5.09 3.02 4.08 1.08 0.  ]\n",
            " [6.01 2.08 4.   1.03 0.  ]\n",
            " [6.03 2.05 4.09 1.05 0.  ]\n",
            " [6.01 2.08 4.07 1.02 0.  ]\n",
            " [6.04 2.09 4.03 1.03 0.  ]\n",
            " [6.06 3.   4.04 1.04 0.  ]\n",
            " [6.08 2.08 4.08 1.04 0.  ]\n",
            " [6.07 3.   5.   1.07 0.  ]\n",
            " [6.   2.09 4.05 1.05 0.  ]\n",
            " [5.07 2.06 3.05 1.   0.  ]\n",
            " [5.05 2.04 3.08 1.01 0.  ]\n",
            " [5.05 2.04 3.07 1.   0.  ]\n",
            " [5.08 2.07 3.09 1.02 0.  ]\n",
            " [6.   2.07 5.01 1.06 0.  ]\n",
            " [5.04 3.   4.05 1.05 0.  ]\n",
            " [6.   3.04 4.05 1.06 0.  ]\n",
            " [6.07 3.01 4.07 1.05 0.  ]\n",
            " [6.03 2.03 4.04 1.03 0.  ]\n",
            " [5.06 3.   4.01 1.03 0.  ]\n",
            " [5.05 2.05 4.   1.03 0.  ]\n",
            " [5.05 2.06 4.04 1.02 0.  ]\n",
            " [6.01 3.   4.06 1.04 0.  ]\n",
            " [5.08 2.06 4.   1.02 0.  ]\n",
            " [5.   2.03 3.03 1.   0.  ]\n",
            " [5.06 2.07 4.02 1.03 0.  ]\n",
            " [5.07 3.   4.02 1.02 0.  ]\n",
            " [5.07 2.09 4.02 1.03 0.  ]\n",
            " [6.02 2.09 4.03 1.03 0.  ]\n",
            " [5.01 2.05 3.   1.01 0.  ]\n",
            " [5.07 2.08 4.01 1.03 0.  ]\n",
            " [6.03 3.03 6.   2.05 1.  ]\n",
            " [5.08 2.07 5.01 1.09 1.  ]\n",
            " [7.01 3.   5.09 2.01 1.  ]\n",
            " [6.03 2.09 5.06 1.08 1.  ]\n",
            " [6.05 3.   5.08 2.02 1.  ]\n",
            " [7.06 3.   6.06 2.01 1.  ]\n",
            " [4.09 2.05 4.05 1.07 1.  ]\n",
            " [7.03 2.09 6.03 1.08 1.  ]\n",
            " [6.07 2.05 5.08 1.08 1.  ]\n",
            " [7.02 3.06 6.01 2.05 1.  ]\n",
            " [6.05 3.02 5.01 2.   1.  ]\n",
            " [6.04 2.07 5.03 1.09 1.  ]\n",
            " [6.08 3.   5.05 2.01 1.  ]\n",
            " [5.07 2.05 5.   2.   1.  ]\n",
            " [5.08 2.08 5.01 2.04 1.  ]\n",
            " [6.04 3.02 5.03 2.03 1.  ]\n",
            " [6.05 3.   5.05 1.08 1.  ]\n",
            " [7.07 3.08 6.07 2.02 1.  ]\n",
            " [7.07 2.06 6.09 2.03 1.  ]\n",
            " [6.   2.02 5.   1.05 1.  ]\n",
            " [6.09 3.02 5.07 2.03 1.  ]\n",
            " [5.06 2.08 4.09 2.   1.  ]\n",
            " [7.07 2.08 6.07 2.   1.  ]\n",
            " [6.03 2.07 4.09 1.08 1.  ]\n",
            " [6.07 3.03 5.07 2.01 1.  ]\n",
            " [7.02 3.02 6.   1.08 1.  ]\n",
            " [6.02 2.08 4.08 1.08 1.  ]\n",
            " [6.01 3.   4.09 1.08 1.  ]\n",
            " [6.04 2.08 5.06 2.01 1.  ]\n",
            " [7.02 3.   5.08 1.06 1.  ]\n",
            " [7.04 2.08 6.01 1.09 1.  ]\n",
            " [7.09 3.08 6.04 2.   1.  ]\n",
            " [6.04 2.08 5.06 2.02 1.  ]\n",
            " [6.03 2.08 5.01 1.05 1.  ]\n",
            " [6.01 2.06 5.06 1.04 1.  ]\n",
            " [7.07 3.   6.01 2.03 1.  ]\n",
            " [6.03 3.04 5.06 2.04 1.  ]\n",
            " [6.04 3.01 5.05 1.08 1.  ]\n",
            " [6.   3.   4.08 1.08 1.  ]\n",
            " [6.09 3.01 5.04 2.01 1.  ]\n",
            " [6.07 3.01 5.06 2.04 1.  ]\n",
            " [6.09 3.01 5.01 2.03 1.  ]\n",
            " [5.08 2.07 5.01 1.09 1.  ]\n",
            " [6.08 3.02 5.09 2.03 1.  ]\n",
            " [6.07 3.03 5.07 2.05 1.  ]\n",
            " [6.07 3.   5.02 2.03 1.  ]\n",
            " [6.03 2.05 5.   1.09 1.  ]\n",
            " [6.05 3.   5.02 2.   1.  ]\n",
            " [6.02 3.04 5.04 2.03 1.  ]\n",
            " [5.09 3.   5.01 1.08 1.  ]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Membagi Dataset => 80 baris data untuk training dan 20 baris data untuk testing\n",
        "dataTraining = np.concatenate((irisDataset[0:40, :], irisDataset[50:90, :]),\n",
        "                              axis=0)\n",
        "dataTesting = np.concatenate((irisDataset[40:50, :], irisDataset[90:100, :]),\n",
        "                             axis=0)"
      ],
      "metadata": {
        "id": "ugU-QiQlezIN"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(dataTesting)\n",
        "len(dataTesting)"
      ],
      "metadata": {
        "id": "DEM0TsHBgMcz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Mendefinisikan Decision Tree Classifier\n",
        "model = tree.DecisionTreeClassifier()"
      ],
      "metadata": {
        "id": "-HDw4xuCgrT1"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Memecah Dataset ke Input dan Label\n",
        "inputTraining = dataTraining[:, 0:4]\n",
        "inputTesting = dataTesting[:, 0:4]\n",
        "labelTraining = dataTraining[:,4]\n",
        "labelTesting = dataTesting[:,4]\n",
        "print(labelTraining)\n",
        "len(labelTraining)"
      ],
      "metadata": {
        "id": "FMb8R-htg3Gn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Mendefinisikan Decision Tree Classifier\n",
        "model = tree.DecisionTreeClassifier()"
      ],
      "metadata": {
        "id": "bCqf9NOzigpl"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Mentraining Model\n",
        "model = model.fit(inputTraining, labelTraining)"
      ],
      "metadata": {
        "id": "J4gFsjTiio_Z"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Memprediksi Input Data Testing\n",
        "hasilPrediksi = model.predict(inputTesting)\n",
        "print(\"Label Sebenarnya : \", labelTesting)\n",
        "print(\"Hasil prediksi : \", hasilPrediksi)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ef8eQd1Yi1Gu",
        "outputId": "e84703e5-2d11-4793-9f1a-881279bed3bb"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Label Sebenarnya :  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]\n",
            "Hasil prediksi :  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Menghitung Akurasi\n",
        "prediksiBenar = (hasilPrediksi == labelTesting).sum()\n",
        "prediksiSalah = (hasilPrediksi != labelTesting).sum()\n",
        "print(\"Prediksi Benar :\", prediksiBenar, \"data\")\n",
        "print(\"Prediksi Salah :\", prediksiSalah, \"data\")\n",
        "print(\"Akurasi :\", prediksiBenar/(prediksiBenar+prediksiSalah) * 100, \"%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KQBfmUh_jXxq",
        "outputId": "c0a300eb-7514-4f46-bcc1-5893039c587f"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Prediksi Benar : 20 data\n",
            "Prediksi Salah : 0 data\n",
            "Akurasi : 100.0 %\n"
          ]
        }
      ]
    }
  ]
}