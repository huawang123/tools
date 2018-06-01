def  has_hz(contents):
    import re
    Pattern = re.compile(u'[\u4e00-\u9fa5]+')
    match = Pattern.search(contents)
    if match:
        return True
    else:
        return False
