



def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    try:
      
      return [resp for resp in resps if resp and resp.strip() and \
              '<Modified_Problem>' in resp and '</Modified_Problem>' in resp and \
              '<The_Value_of_New_Variable2>' in resp and '</The_Value_of_New_Variable2>' in resp and \
                len(resp.split('<Modified_Problem>')[-1].split('</Modified_Problem>')[0]) >= len(data['data2']['problem']) ]
    
    except Exception as e:
      print('filter func error:', e)
      raise e

