
def article(word):
    if word.strip()[0].lower() in ["a", "e", "i", "o", "u"]:
        return "An"
    else:
        return "A"


def describe(thresholds, words, value):
    """
    
    thresholds = lower bound of each word in descending order\n
    len(words) = len(thresholds) + 1
    """
    
    # words = ["implies a seriously increased risk", "implies an increased risk", "implies a small increase in risk", "does not significantly effect the risk",
    #           "implies slightly smaller risk", "implies a decreased risk", "implies a greatly decreased risk"]
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    
    i = 0
    
    while i < len(thresholds) and value > thresholds[i]:
        
        # print(value, "greater than", thresholds[i])
        i = i + 1
    # print( words[i])
    return words[i]

# [-1.89, -1.26, -0.63, 0.63, 1.26, 1.89]
# Format the metrics for display and descriptions
def format_metric(metric):
    return metric.replace("_", " ").capitalize()


# look up formated metric name for display and descriptions
def lookup_metric(metric, parameters):
    # for column name look up Explanation in the parameters
    explanation = parameters.loc[parameters['Parameter'] == metric, 'Explanation'].values[0]
    explanation.strip().capitalize()
    return explanation

def describe_contributions(value,  thresholds = [10, 5, 2, -2,-5,-10], words = ["implies a seriously increased risk", "implies an increased risk", "implies a small increase in risk", "does not significantly effect the risk", "implies slightly smaller risk", "implies a decreased risk", "implies a greatly decreased risk"]):
    # words=words.reverse()   
    return describe(thresholds, words, value)

def format_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    else:
        return round(value, 2)
    