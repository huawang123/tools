import pinyin

def to_pinyin(var_str):
    """ 汉字[钓鱼岛是中国的]=>拼音[diaoyudaoshizhongguode]\n
    汉字[我是shui]=>拼音[woshishui]\n
    汉字[AreYou好]=>拼音[AreYouhao]\n
    汉字[None]=>拼音[]\n 汉字[]=>拼音[]\n
    :param var_str: str 类型的字符串
    :return: 汉字转小写拼音 """
    if isinstance(var_str, str):
        if var_str == 'None':
            return ""
        else:
            return pinyin.get(var_str, format='strip', delimiter="")
    else:
        return '类型不对'