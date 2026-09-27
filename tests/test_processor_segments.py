from app.core.timeline import apply_preferences, union

def test_union_sorts_and_does_not_shrink_contained_intervals():
    assert union([{'start':30,'end':60},{'start':35,'end':40},{'start':5,'end':10}]) == [{'start':5,'end':10},{'start':30,'end':60}]

def test_only_selected_categories_are_removed():
    rows=[{'start':0,'end':5,'label':'Ad'},{'start':5,'end':10,'label':'Content'},{'start':10,'end':15,'label':'Promo'}]
    assert [(r['start'],r['end']) for r in apply_preferences(rows,{'remove_ads':True})['segments']] == [(0,5)]

def test_editorial_category_cannot_be_directly_selected():
    assert apply_preferences([{'start':0,'end':5,'label':'EditorialNonSpeech'}],{'remove_editorial_non_speech':True})['segments']==[]

def test_short_island_policy_is_explicit_and_can_be_disabled():
    rows=[{'start':0,'end':5,'label':'Ad'},{'start':5,'end':8,'label':'Content'},{'start':8,'end':15,'label':'Ad'}]
    assert len(apply_preferences(rows,{'remove_ads':True,'minimum_retained_seconds':0})['segments'])==2
    result=apply_preferences(rows,{'remove_ads':True})
    assert [(r['start'],r['end']) for r in result['segments']]==[(0,15)]
    assert 'short_island' in result['segments'][0]['sources']
