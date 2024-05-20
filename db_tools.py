# from matplotlib.style import available
import psycopg2
from sqlalchemy import create_engine,text # inspect
import pandas as pd
import json,os,glob
import numpy

from config import *

warning = []

def read_source_data(db_name,sql_text):
    #print(sql_text)
    db_eng = create_engine(available_dbs[db_name][0]) 
    connection = db_eng.raw_connection()
    sql_df = pd.read_sql(sql_text ,con=connection)
    connection.close()
    #print(sql_df.shape)
    return(sql_df)

def read_target(target_db,table):
    #print(sql_text)
    db_eng = create_engine(target_dbs[target_db][0]) 
    sql_text = """
    SELECT * from {}.{};
    """.format(target_dbs[target_db][1],table)    
    connection = db_eng.raw_connection()
    sql_df = pd.read_sql(sql_text ,con=connection)
    connection.close()
    #print(sql_df.shape)
    return(sql_df)


def write_target(sql_df,target_db,table):
    #print(sql_text)
    db_eng = create_engine(target_dbs[target_db][0]) 
    connection = db_eng.raw_connection()
    sql_df.head(1000).to_sql( table,connection,schema= target_dbs[target_db][1], if_exists = 'replace', index=False,chunksize = 100000)
    connection.close()
    #print(sql_df.shape)



def get_tables (db_name, text_file_location=None):
 
#    sql_df = pd.read_sql(
#        "select * from mimiciii.patients limit 10",
#        con=db_eng
#    )
 
    def convert(o):
        if not isinstance(o, str): return str(o)  
        raise TypeError

    tables = []

    if "TEXT"  not in db_name:
        db_eng = create_engine(available_dbs[db_name][0]) 

        #inspector = inspect(db_eng)
        #schemas = inspector.get_schema_names()
        tables = []
        schema = available_dbs[db_name][1]
    #    for table_name in inspector.get_table_names(schema=schema):
    #        columns ={}
    #        for column in inspector.get_columns(table_name, schema=schema):
    #            columns [column['name']] = column
    #        tables.append(table_name)
        sql_text = """
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = '{}';
        """.format(schema)
        connection = db_eng.raw_connection()
        sql_df = pd.read_sql( sql_text,con=connection)
        print(sql_df.shape)
        tables = sql_df.table_name.to_list()
        connection.close()  
    else:    
        if os.path.exists(text_file_location) :
            old_path = os.getcwd()
            extension = 'csv'
            os.chdir(text_file_location)
            tables = glob.glob('*.{}'.format(extension))
            os.chdir(old_path)   
 
 
    return  tables
             
def get_table_cols (db_name,table_name):
 
#    sql_df = pd.read_sql(
#        "select * from mimiciii.patients limit 10",
#        con=db_eng
#    )
 
    def convert(o):
        if not isinstance(o, str): return str(o)  
        raise TypeError
        
    db_eng = create_engine(available_dbs[db_name][0]) 

    #inspector = inspect(db_eng)
    #schemas = inspector.get_schema_names()
    tables = []
    schema = available_dbs[db_name][1]
#    for table_name in inspector.get_table_names(schema=schema):
#        columns ={}
#        for column in inspector.get_columns(table_name, schema=schema):
#            columns [column['name']] = column
#        tables.append(table_name)
    sql_text = """
    SELECT column_name 
    FROM information_schema.columns
    WHERE table_schema = '{}' and table_name = '{}';
    """.format(schema,table_name)

    connection = db_eng.raw_connection()  
    sql_df = pd.read_sql( sql_text,con=connection)
    print(sql_df.shape)
    columns = sql_df.column_name.to_list()
    connection.close()
    return  columns

def get_table_profile (db_name,table_name,samplesize,text_folder=None):
 
    print('getting profile -- ', db_name,table_name,samplesize,text_folder)
 
    global warning

    import dataprofiler as dp

    profile_options = dp.ProfilerOptions()
    profile_options.structured_options.text.is_enabled = False
    profile_options.structured_options.text.vocab.is_enabled = True
    profile_options.structured_options.int.variance.is_enabled = True
    profile_options.structured_options.data_labeler.is_enabled = False

    if not "TEXT" in db_name:
    
        db_eng = create_engine(available_dbs[db_name][0]) 
        connection = db_eng.raw_connection()
        profile_df = pd.read_sql(
            "select * from {}.{} order by random() limit {}".format(available_dbs[db_name][1],table_name,samplesize),
            con=connection
        )
        connection.close()
    else:
        profile_df=pd.read_csv(os.path.join(text_folder,table_name))
        profile_df=profile_df.sample(n=samplesize,replace=True)

    print(profile_df.shape)


    if len(profile_df) == 0:
        warning.append(f'No data found in the table {table_name}')
        return None 
    else:
        if len(profile_df) <1000:
            warning.append(f'Data size is too small ( {len(profile_df)}) for {table_name} to generate profile, the final result may not be accurate.')

        profile_df.to_csv('./data_profile/table_{}_sample.csv'.format(table_name),index=False)
        
        profile = dp.Profiler(profile_df,samples_per_update = samplesize,min_true_samples=samplesize,options= profile_options)
        report = profile.report(report_options={"output_format":"pretty"})

        with open('./data_profile/table_{}_profile.json'.format(table_name), 'w') as f:
            json.dump(report, f)

        return  report
                            


# print (get_tables ('PSQL_MIMIC'))
# result = get_table_profile('PSQL_MIMIC','patients',100) 
# print(result)
# table_mapping={}
# for x in result['data_stats']:
#     print(x)
#     if x['categorical'] :
#         table_mapping[x['column_name']]=x['statistics']['categories']
#     else:    
#         table_mapping[x['column_name']]=[]

# print(json.dumps(table_mapping,indent= 4 ))   

#for x in result['data_stats']:
#    print(x)

 

def process_mapped_tables(selected_mappings):

    mapped_tables = {}

    for x in selected_mappings:
        if (len(x[0]) != len(x[1])) or len(x[0]) < 2:
            print( x, '  - invalid mapping')
            continue
        else:
            # print(x)
            if (x[1][0][0],x[0][0][0]) not in mapped_tables.keys() :
                mapped_tables[(x[1][0][0],x[0][0][0]) ] = {}
            if x[1][1][0] not in mapped_tables[(x[1][0][0],x[0][0][0]) ].keys() :
                mapped_tables[(x[1][0][0],x[0][0][0]) ][x[1][1][0]] = x[0][0][0]+'.'+x[0][1][0]
 
    return mapped_tables

def process_mapped_fields(selected_mappings):

    mapped_table_files = {}

    for x in selected_mappings:
        if (len(x[0]) != len(x[1])) or len(x[0]) < 2:
            print( x, '  - invalid mapping')
            continue
        else:
            # print(x)
            if len(x[0]) == 3:
                if x[0][0][0] not in mapped_table_files.keys() :
                    mapped_table_files[x[0][0][0]] = {}
                if x[0][1][0] not in mapped_table_files[x[0][0][0]].keys() :
                    mapped_table_files[x[0][0][0]][x[0][1][0]]={}
                    mapped_table_files[x[0][0][0]][x[0][1][0]]['target'] = x[1][0][0]+'.'+x[1][1][0]
                    mapped_table_files[x[0][0][0]][x[0][1][0]]['maps'] ={}
                mapped_table_files[ x[0][0][0]][x[0][1][0]]['maps'][x[0][2][0]] = x[1][2][0]
    return mapped_table_files

def read_mapped_statistics(selected_mappings,source_db):    

    mapped_table_files = process_mapped_fields(selected_mappings)

    table_count_query = """
    select '{table}' as "source table" ,'' as "source field", '{target}' as target,'' as "source value",'' as "target value", count(*) from {schema}.{table} ;
    """

    field_count_query = """
    select '{table}'as "source table" , '{field}' as "source field",'{target}' as target, {field} as "source value" ,'' as "target value", count(*) from {schema}.{table} where {field} in ({field_value_list} ) group by {field};
    """
    non_exists_field_count_query ="""
    select '{table}' as "source table" ,'{field}' as "source field", '{target}' as target, {field}  as "source value" ,'HUDHNJDN233' as "target value", count(*) from {schema}.{table}  where {field} not in ({field_value_list} ) group by {field};
    """
    db_eng = create_engine(source_db[0]) 
    db_conn= db_eng.connect()
    result_set = []
    for t in mapped_table_files.keys():
        for fe in mapped_table_files[t]:
            query_texts =[]
            # map_df = pd.DataFrame(mapped_table_files[t][fe]['maps'].items(), columns=['source value', 'target value'])
            # query_texts.append('select * from mimiciii.patients limit 10')
            query_texts.append(table_count_query.format(table=t,schema=source_db[1],target = mapped_table_files[t][fe]['target'].split('.')[0]))
            query_texts.append(field_count_query.format(table=t,schema=source_db[1],field = fe,field_value_list = ', '.join(['\'{}\''.format(value) for value in mapped_table_files[t][fe]['maps'].keys()]),target = mapped_table_files[t][fe]['target']))
            query_texts.append(non_exists_field_count_query.format(table=t,schema=source_db[1],field = fe,field_value_list = ', '.join(['\'{}\''.format(value) for value in mapped_table_files[t][fe]['maps'].keys()]),target = mapped_table_files[t][fe]['target']))
            for query_text in query_texts:
                # query_result = db_conn.execute(query_text)
                # print(type(query_result))
                # columns = [desc[0] for desc in query_result.description]
                # df = pd.DataFrame(query_result.fetchall(), columns=columns)
                df = pd.read_sql(query_text,con=db_conn)
                if len(df) > 0:
                    df['target value'] = df['source value'].map(mapped_table_files[t][fe]['maps'])                 
                    result_set.append(df)
    
    db_conn.close()
    mapping_validation_result = pd.concat(result_set, axis=0).reset_index(drop=True)
    mapping_validation_result['target value'] = mapping_validation_result.apply(lambda x: x['source value'] if x['source value']=='' else x['target value'], axis=1)
    mapping_validation_result['target value']  = mapping_validation_result['target value'].fillna('Not Mapped') 
    # print(mapping_validation_result)
    return mapping_validation_result



