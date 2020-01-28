import xml.dom.minidom
import pymysql
import os
import re
def find_child(Par_nodes, mystr):
    for child_node in Par_nodes:
        if(len(child_node.childNodes) > 0):
            mystr = find_child(child_node.childNodes, mystr)
        elif(child_node.nodeValue != None):
            mystr += child_node.data.replace('\n', '')
    return mystr

class myfile:
    def __init__(self):
        self.name = ""

def search(folder, myfilter, allfile):
    folders = os.listdir(folder)
    for name in folders:
        curname = os.path.join(folder, name)
        isfile = os.path.isfile(curname)
        if isfile:
            ext = os.path.splitext(curname)[1]
            count = myfilter.count(ext)
            if count > 0:
                cur = myfile()
                cur.name = curname
                allfile.append(cur.name)
        else:
            search(curname, myfilter, allfile)
    return allfile

if __name__ == '__main__':
    folder = r"D:\专利数据\cc-TXTS-10-B 中国发明专利授权公告标准化全文文本数据"
    filter = [".XML"]
    allfile = []
    allfile = search(folder, filter, allfile)
    file_len = len(allfile)
    patt_techField = '技术领域'
    patt_techBg = '背景技术'
    patt_content = '发明内容'
    patt_pic = '附图说明'
    patt_detail = '具体实施方式'
    # patt_rfr = 'F25D'
    # patt_washing = 'D06F'
    # patt_cooler = 'F24F'
    #patt_charge = 'B60L 11'
    #patt_transport = 'G08G 1'
    print('共查找到%d个.XML文件' %(file_len))
    db = pymysql.connect("localhost", "root", "", "patent_system")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    patent_id = 0    #################################################################
    for num in range(file_len):    #################################################################
        xml_name = allfile[num]    ####################################################
        #xml_name = r'D:\专利数据\cc-TXTS-10-B 中国发明专利授权公告标准化全文文本数据\20180525\1\CN112011000076153CN00001040250410BFULZH20180525CN005\CN112011000076153CN00001040250410BFULZH20180525CN005.XML'
        #xml_name = r'3.XML'
        print('XML文件名称：%s' %(xml_name))
        search_CN = re.search('CN', xml_name)
        if search_CN:
            dom1 = xml.dom.minidom.parse(xml_name)  # 打开xml文件
            root = dom1.documentElement  # 得到文档元素对象
            ipc_pars = root.getElementsByTagName('business:ClassificationIPCR')
            # for i in range(len(ipc_pars)):
            # for i in range(1):
            str_ipc = ipc_pars[0].getElementsByTagName('base:Text')[0].firstChild.data
            if len(str_ipc.split('    ')) == 2:
                str_ipc = str_ipc.split('    ')[0]
            elif len(str_ipc.split('    ')) == 3:
                str_ipc = '   '.join(str_ipc.split('    ')[:2])
            # print(str_ipc)
            # search_rfr = re.search(patt_rfr, str_ipc)
            # search_washing = re.search(patt_washing, str_ipc)
            # search_cooler = re.search(patt_cooler, str_ipc)
            # if search_rfr or search_washing or search_cooler:
            #if search_washing:
            #if search_charge:
            #if search_transport:
            app_nums = root.getElementsByTagName('base:DocNumber')  # 按标签名称查找，返回标签结点数组
            app_num = app_nums[2]
            str_appNum = app_num.firstChild.data
            print('专利申请号：' + str_appNum)
            titles = root.getElementsByTagName('business:InventionTitle')
            title = titles[0]
            str_title = title.firstChild.data
            print('专利名称：' + str_title)
            Paragraphs = root.getElementsByTagName('base:Paragraphs')
            abstract = Paragraphs[0]
            str_abstract = abstract.firstChild.data
            print('专利摘要：' + str_abstract)
            company_names = root.getElementsByTagName('base:Name')
            company_name = company_names[0]
            str_comName = company_name.firstChild.data
            print('公司名称：' + str_comName)
            str_tecField = ''
            str_tecBg = ''
            str_content = ''
            par_num = 1
            # 获取技术领域
            while True:
                if (par_num > len(Paragraphs) - 1):
                    break
                if (Paragraphs[par_num].firstChild == None):
                    par_num += 1
                    continue
                elif (Paragraphs[par_num].firstChild.nodeValue != None):
                    search_techField = re.search(patt_techField, Paragraphs[par_num].firstChild.data)
                    if (search_techField):
                        while True:
                            par_num += 1
                            if (par_num > len(Paragraphs) - 1):
                                break
                            if (Paragraphs[par_num].firstChild == None):
                                continue
                            search_techBg = re.search(patt_techBg, Paragraphs[par_num].firstChild.data)
                            search_content = re.search(patt_content, Paragraphs[par_num].firstChild.data)
                            search_pic = re.search(patt_pic, Paragraphs[par_num].firstChild.data)
                            search_detail = re.search(patt_detail, Paragraphs[par_num].firstChild.data)
                            if (search_techBg or search_content or search_pic or search_detail):
                                break
                            str_tecField = find_child(Paragraphs[par_num].childNodes, str_tecField)
                        break
                    else:
                        par_num += 1
                else:
                    par_num += 1
            print('技术领域：' + str_tecField)

            # 获取技术背景
            while True:
                if (par_num > len(Paragraphs) - 1):
                    break
                if (Paragraphs[par_num].firstChild == None):
                    par_num += 1
                    continue
                elif (Paragraphs[par_num].firstChild.nodeValue != None):
                    search_techBg = re.search(patt_techBg, Paragraphs[par_num].firstChild.data)
                    if (search_techBg):
                        while True:
                            par_num += 1
                            if (par_num > len(Paragraphs) - 1):
                                break
                            if (Paragraphs[par_num].firstChild == None):
                                continue
                            search_content = re.search(patt_content, Paragraphs[par_num].firstChild.data)
                            search_pic = re.search(patt_pic, Paragraphs[par_num].firstChild.data)
                            search_detail = re.search(patt_detail, Paragraphs[par_num].firstChild.data)
                            if (search_content or search_pic or search_detail):
                                break
                            str_tecBg = find_child(Paragraphs[par_num].childNodes, str_tecBg)
                        break
                    else:
                        par_num += 1
                        continue
                else:
                    par_num += 1
            print('背景技术：' + str_tecBg)

            # 获取发明内容
            while True:
                if (par_num > len(Paragraphs) - 1):
                    break
                if (Paragraphs[par_num].firstChild == None):
                    par_num += 1
                    continue
                elif (Paragraphs[par_num].firstChild.nodeValue != None):
                    search_content = re.search(patt_content, Paragraphs[par_num].firstChild.data)
                    if (search_content):
                        while True:
                            par_num += 1
                            if (par_num > len(Paragraphs) - 1):
                                break
                            if (Paragraphs[par_num].firstChild == None):
                                continue
                            search_pic = re.search(patt_pic, Paragraphs[par_num].firstChild.data)
                            search_detail = re.search(patt_detail, Paragraphs[par_num].firstChild.data)
                            if (search_pic or search_detail):
                                break
                            str_content = find_child(Paragraphs[par_num].childNodes, str_content)
                        break
                    else:
                        par_num += 1
                        continue
                else:
                    par_num += 1
            print('发明内容：' + str_content)

            if (str_tecField != '' and str_tecBg != '' and str_content != ''):
                patent_id += 1
                my_dict = {}
                my_dict['id'] = patent_id
                my_dict['app_num'] = pymysql.escape_string(str_appNum)
                my_dict['title'] = pymysql.escape_string(str_title)
                my_dict['abstract'] = pymysql.escape_string(str_abstract)
                my_dict['company_name'] = pymysql.escape_string(str_comName)
                my_dict['content'] = pymysql.escape_string(str_content)
                my_dict['tech_field'] = pymysql.escape_string(str_tecField)
                my_dict['tech_bg'] = pymysql.escape_string(str_tecBg)
                my_dict['label'] = pymysql.escape_string(str_ipc)

                # SQL 插入语句
                sql = """INSERT INTO tb_patentall_label(id, app_num, title, abstract, company_name, content, tech_field, tech_bg, label)
                    VALUES ({id}, '{app_num}', '{title}', '{abstract}', '{company_name}', '{content}', '{tech_field}', '{tech_bg}', '{label}')""".format(**my_dict)

                try:
                    # 执行sql语句
                    cursor.execute(sql)
                    # 提交到数据库执行
                    db.commit()
                    print('第%d条数据录入成功！' %(patent_id))
                    print('***********************************************************************************************************')
                except IndexError as e:
                    # 如果发生错误则回滚
                    db.rollback()
                    print(e)
                    print('第%d条数据录入失败！' %(patent_id))
    # 关闭数据库连接
    db.close()
