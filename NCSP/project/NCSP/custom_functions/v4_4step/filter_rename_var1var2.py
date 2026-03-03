



def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    try:
      
      res = [resp for resp in resps if resp and resp.strip() and \
              '<The_Symbol_of_New_Variable1>' in resp and '</The_Symbol_of_New_Variable1>' in resp and \
              '<The_Symbol_of_New_Variable2>' in resp and '</The_Symbol_of_New_Variable2>' in resp]
      
      # print(resps, 'filter res', res)
      return res
    
    except Exception as e:
      print('filter func error:', e)
      raise e

