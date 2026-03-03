



def main(data:dict, resps:list, finish_reasons:list, stop_reasons:list) -> list:
    try:
      
      return [resp for resp in resps if resp and resp.strip() and \
              '<The_Value_of_New_Variable2>' in resp and '</The_Value_of_New_Variable2>' in resp ]
    
    except Exception as e:
      print('filter func error:', e)
      raise e

