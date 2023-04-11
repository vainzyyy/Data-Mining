{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPSqDgzM3xXqTzco6KqqOC3",
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
        "<a href=\"https://colab.research.google.com/github/nabiltoby/Aplikasi-Biodata/blob/main/Preprocessing.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cfARDoKq_Ezn"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Numpy merupakan library python digunakan untuk komputas matriks\n",
        "Matplotlib merupakan library python untuk presentasi data berupa grafik atau plot"
      ],
      "metadata": {
        "id": "RNElERGsCHza"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset = pd.read_csv('Data2.csv')\n",
        "x = dataset.iloc[:, :-1].values\n",
        "y = dataset.iloc[:, -1].values\n"
      ],
      "metadata": {
        "id": "O-Amt-6MCeJO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V_AP-J1_D39L",
        "outputId": "9f4ffb30-f87b-4358-e458-d4962baad253"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 24.0 45000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 nan]\n",
            " ['Germany' nan 58000.0]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' nan 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 60.0 nan]\n",
            " ['France' 50.0 67000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3GsNJ6ayEV6B",
        "outputId": "fbb69de3-c112-4cfd-ad0f-75c94b9959bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['No' 'Yes' 'No' 'No' 'Yes' 'No' 'Yes' 'No' 'No' 'No']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.impute import SimpleImputer\n",
        "imputer = SimpleImputer(missing_values=np.nan, strategy='mean')\n",
        "imputer.fit(x[:, 1:3])\n",
        "x[:, 1:3] = imputer.transform(x[:, 1:3])"
      ],
      "metadata": {
        "id": "suz8xmxoElSF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bkhr7_oiHP5w",
        "outputId": "2b43e694-fa5b-4836-b869-99669d0a3d73"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 24.0 45000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 60625.0]\n",
            " ['Germany' 41.125 58000.0]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' 41.125 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 60.0 60625.0]\n",
            " ['France' 50.0 67000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "ct = ColumnTransformer(transformers=[('encider', OneHotEncoder(), [0])], remainder='passthrough')\n",
        "x = np.array(ct.fit_transform(x))"
      ],
      "metadata": {
        "id": "Gssj8z8bFKrM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zdRdbuKcHh0_",
        "outputId": "c48ad59d-f4dd-4c59-d172-9db664ed0343"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1.0 0.0 0.0 44.0 72000.0]\n",
            " [0.0 0.0 1.0 24.0 45000.0]\n",
            " [0.0 1.0 0.0 30.0 54000.0]\n",
            " [0.0 0.0 1.0 38.0 60625.0]\n",
            " [0.0 1.0 0.0 41.125 58000.0]\n",
            " [1.0 0.0 0.0 35.0 58000.0]\n",
            " [0.0 0.0 1.0 41.125 52000.0]\n",
            " [1.0 0.0 0.0 48.0 79000.0]\n",
            " [0.0 1.0 0.0 60.0 60625.0]\n",
            " [1.0 0.0 0.0 50.0 67000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "le = LabelEncoder()\n",
        "y = le.fit_transform(y)"
      ],
      "metadata": {
        "id": "R46Lyqx5JXNe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print (y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Hr1YWmHIJpJo",
        "outputId": "8c627ab3-d04f-46ff-ff13-c24a506ba45d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 1 0 0 1 0 1 0 0 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)"
      ],
      "metadata": {
        "id": "VFNRGiwZKFDD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hltpXHkRK1Y7",
        "outputId": "309d008e-6a07-40b7-833d-a561e91e80d2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.0 0.0 1.0 41.125 52000.0]\n",
            " [0.0 1.0 0.0 41.125 58000.0]\n",
            " [1.0 0.0 0.0 44.0 72000.0]\n",
            " [0.0 0.0 1.0 38.0 60625.0]\n",
            " [0.0 0.0 1.0 24.0 45000.0]\n",
            " [1.0 0.0 0.0 48.0 79000.0]\n",
            " [0.0 1.0 0.0 60.0 60625.0]\n",
            " [1.0 0.0 0.0 35.0 58000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ct5W3ODQLTxS",
        "outputId": "9f8bcf7a-41e5-4c2d-a651-e908abeeb33e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.0 1.0 0.0 30.0 54000.0]\n",
            " [1.0 0.0 0.0 50.0 67000.0]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VK7Wv9nwLcD6",
        "outputId": "0bb681c1-7875-493f-adfb-30e384941fd5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1 1 0 0 1 0 0 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cBeeOp2fLhFq",
        "outputId": "2e7d17ba-ecf5-4424-b8f2-32d756b436cf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "sc = StandardScaler()\n",
        "x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])\n",
        "x_test[:, 3:] = sc.transform(x_test[:, 3:])"
      ],
      "metadata": {
        "id": "va1hZqU8L5Pl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-4kHFmdsNKsd",
        "outputId": "6dd9e23a-7e30-4d5d-ff93-f5235136fb94"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.0 0.0 1.0 -0.02901429951384512 -0.8659844920088038]\n",
            " [0.0 1.0 0.0 -0.02901429951384512 -0.2657353134323044]\n",
            " [1.0 0.0 0.0 0.26757631773879387 1.134846103246194]\n",
            " [0.0 0.0 1.0 -0.3513954052232353 -0.0031262978050859342]\n",
            " [0.0 0.0 1.0 -1.7956627588013034 -1.566275200348053]\n",
            " [1.0 0.0 0.0 0.6802241330468134 1.8351368115854434]\n",
            " [0.0 1.0 0.0 1.9181675789708719 -0.0031262978050859342]\n",
            " [1.0 0.0 0.0 -0.66088126670425 -0.2657353134323044]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QCqbKIbqNhmp",
        "outputId": "64a954fc-806e-4dca-c93b-8bbed6ce667a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.0 1.0 0.0 -1.1766910358392744 -0.6659014324833039]\n",
            " [1.0 0.0 0.0 0.886548040700823 0.6346384544324446]]\n"
          ]
        }
      ]
    }
  ]
}