# col = 'Department'

def company_review_sectionwise(col,review)
    for index,section in enumerate(leap_data[col].unique()):
        text = ' '.join(list(leap_data[leap_data[col]==section][review]))
        result = prediction_result(text)
        ax = plt.subplot(1, len(leap_data[col].unique()), index+1)
        ax.bar(['negative','Positive'], result[1])
        ax.set_xlabel(section)
        yield ax