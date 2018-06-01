
str1 = list(filter(lambda x: x.isdigit(), str0))
#……function（.isdigit()）可以变通查询字符串比如‘aaaa’.后边弹出的属性
str1 = list(filter(lambda x: x not in '0123456789', str0))

str = ''.join(str1)
