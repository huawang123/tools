def del_left_num(s):
    for c in s:
        if c.isdigit():
            s = s[1:]
        else:
            break
    return s
