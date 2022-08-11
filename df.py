def calc_df(coll):
    """Calculate DF of each term in vocab and return as term:df dictionary."""
    df_ = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try:
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    return df_
