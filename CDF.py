import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ks_2samp

attrib = "GPS_longitude"
"""GPS_longitude
crop_count
age_malehead
"""

df = pd.read_csv("./data/farmer_survey.csv")
df = df.fillna(0)
# getting data of the histogram
count, bins_count = np.histogram(df[attrib], bins=10)

# finding the PDF of the histogram using count values
pdf = count / sum(count)

# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
plt.plot(cdf, bins_count[1:], label="Original data")


df_1 = pd.read_csv("./synthetic_table/farmer_survey_synthetic_with_privacy.csv")
df_1 = df_1.fillna(0)
# getting data of the histogram
count_1, bins_count_1 = np.histogram(df_1[attrib], bins=10)

# finding the PDF of the histogram using count values
pdf_1 = count_1 / sum(count_1)

# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf_1 = np.cumsum(pdf_1)


print(ks_2samp(df[attrib], df_1[attrib]))


# plotting PDF and CDF
plt.plot(cdf_1, bins_count_1[1:], label="Synthetic data")
plt.legend()
plt.ylabel("CDF")
plt.xlabel("Feature value")
plt.title("Attribute with High Privacy Risk")

plt.savefig("./output/cdf/cdf_high.png")