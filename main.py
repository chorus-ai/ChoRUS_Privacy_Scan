# Form implementation generated from reading ui file 'OMOP_ETL.ui'
#
# Created by: PyQt6 UI code generator 6.3.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from pathlib import Path
import json,os

from select_table_to_process import Ui_SelectTableToProcess
from view_phi_result import Ui_ViewPHIScan
from json_data_viwer import Ui_Json_Viewer

from config import *

from phi_scan import phi_scan
import pandas as pd

from db_tools import get_table_profile

class DB_Profile_Thread(QtCore.QThread):
    log = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(DB_Profile_Thread, self).__init__(parent)
        self.status= ''
        self._items = []
        self.source_profile ={}
        self.source_mapping={}
        self.text_folder = ""

    def setItems(self, items,source_db,sampleqty,text_folder=None):
        if not self.isRunning():
            self._items[:] = items
            self._source_db = source_db
            self._sampleqty = sampleqty
            self.text_folder = text_folder

    def run(self):
        for item in self._items:
            self.log.emit('processing:  %s' % item  )
            table_mapping ={}
            if "TEXT" in self._source_db:
                profile_table_result = get_table_profile(self._source_db,item,self._sampleqty,self.text_folder)
            else:                
                profile_table_result = get_table_profile(self._source_db,item,self._sampleqty)

            if profile_table_result is None:            
                self._items = [x for x in self._items if x != item] 
                continue

            self.source_profile[item] = profile_table_result['data_stats']
            for x in profile_table_result['data_stats']:
                if x['categorical'] :
                    table_mapping[x['column_name']]= [x for  x in x['statistics']['categorical_count']] 
                else:    
                    table_mapping[x['column_name']]=[]
            self.source_mapping[item]= table_mapping      
            self.log.emit('finished:  %s' % item  )
        self.source_profile =  json.loads(json.dumps(self.source_profile))  


class PHI_Scan_Thread(QtCore.QThread):
    log = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(PHI_Scan_Thread, self).__init__(parent)
        self.status= ''
        self._items = []
        self.phi_scan_result_json ={}


    def setItems(self, items,source_db):
        if not self.isRunning():
            self._items[:] = items
            self._source_db = source_db

    def run(self):
        for item in self._items:
            self.log.emit('processing:  %s' % item  )
            original_data_path = './data_profile/table_{}_sample.csv'.format(item)
            json_file_path = './data_profile/table_{}_profile.json'.format(item)
            model_path = PHI_SCAN_MODEL
            output_path = './data_profile/table_{}_phi.csv'.format(item)
            print(item)
            phi_scan(original_data_path,json_file_path,model_path,output_path)
            phi_scan_result = pd.read_csv(output_path,index_col=0)          

            for index,row in  phi_scan_result.iterrows():
                self.phi_scan_result_json['{}.{}'.format(item,index)] = {'method':'0','offset':'','predict_probability':row['ML prediction result'],'predict_result':row['ML prediction result 0/1']}

            self.log.emit('finished:  %s' % item  )
        self.phi_scan_result_json =  json.loads(json.dumps(self.phi_scan_result_json))  




# source_tables = ['chartevents', 'inputevents_mv', 'admissions', 'callout', 'caregivers', 'ccs_multi_level_dx', 'ccs_single_level_dx', 'icustays', 'noteevents', 'cptevents', 'd_cpt', 'd_icd_diagnoses', 'd_icd_procedures', 'd_items', 'd_labitems', 'datetimeevents', 'diagnoses_icd', 'inputevents_cv', 'drgcodes', 'gcpt_admission_location_to_concept', 'gcpt_admission_type_to_concept', 'gcpt_admissions_diagnosis_to_concept', 'gcpt_atb_to_concept', 'gcpt_care_site', 'gcpt_chart_label_to_concept', 'gcpt_chart_observation_to_concept', 'gcpt_continuous_unit_carevue', 'gcpt_cpt4_to_concept', 'gcpt_cv_input_label_to_concept', 'gcpt_datetimeevents_to_concept', 'gcpt_derived_to_concept', 'gcpt_discharge_location_to_concept', 'gcpt_drgcode_to_concept', 'gcpt_ethnicity_to_concept', 'gcpt_heart_rhythm_to_concept', 'gcpt_inputevents_drug_to_concept', 'gcpt_insurance_to_concept', 'gcpt_lab_label_to_concept', 'gcpt_lab_unit_to_concept', 'gcpt_lab_value_to_concept', 'gcpt_labs_from_chartevents_to_concept', 'gcpt_labs_specimen_to_concept', 'gcpt_map_route_to_concept', 'gcpt_marital_status_to_concept', 'gcpt_microbiology_specimen_to_concept', 'gcpt_mv_input_label_to_concept', 'gcpt_note_category_to_concept', 'gcpt_note_section_to_concept', 'gcpt_org_name_to_concept', 'gcpt_output_label_to_concept', 'gcpt_prescriptions_ndcisnullzero_to_concept', 'gcpt_procedure_to_concept', 'gcpt_religion_to_concept', 'gcpt_resistance_to_concept', 'gcpt_route_to_concept', 'gcpt_seq_num_to_concept', 'gcpt_spec_type_to_concept', 'gcpt_unit_doseera_concept_id', 'labevents', 'heightfirstday', 'patients', 'outputevents', 'microbiologyevents', 'mimic_140features_adm', 'mimic_140features_ts', 'mimic_feature_dict', 'mimic_general_feature_mapping', 'mimic_ts_feature_mapping', 'prescriptions', 'procedureevents_mv', 'procedures_icd', 'services', 'transfers', 'weightfirstday', 'stroke_icds', 'depression_def']
# source_tables = ['aa']
class Ui_DataMappingTools(object):
    source_tables=[]
    selected_tables_list=[]
    selected_mappings = []
    source_profile = {}
    source_mapping = {}
    target_mapping = {}
    selected_phi_method ={}
    table_mapping = []
    mapped_tables_list = []
    path = None
    generated_SQL = None
    universal_ID={}
    text_folder=''
    
    # with open('mapping_profile.json', 'r') as f:
    #     source_mapping = json.load(f)      
 

    # with open('selected_mappings.json', 'r') as f:
    #     selected_mappings = json.load(f)['mapped_items'] 

   # print(json.dumps(source_mapping,indent=4))      
    def new_project(self):
        self.source_tables=[]
        self.selected_tables_list=[]
        self.selected_mappings = []
        self.source_profile = {}
        self.source_mapping = {}
        self.target_mapping = {}
        self.selected_phi_method ={}
        self.table_mapping = []
        self.mapped_tables_list = []       
        self.generated_SQL= None
        self.universal_ID={}
        # todo - display selected tables  / display seleted mappings         
        self.update_selected_tables()
        self.update_mapped_tables()
        self.text_folder = ''

 # action called by file open action
    def file_open(self):
 
        # getting path and bool value
        path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open file", "",
                             "Json File (*.json)")
 
        # if path is true
        if path:
            # try opening path
            try:
                with open(path, 'r') as f:
                    # read the file
                    saved_json = json.load(f)
 
            # if some error occurred
            except Exception as e:
 
                # show error using critical method
                self.dialog_critical(str(e))
            # else
            else:
                # update path value
                self.path = path

                # update content
                self.selected_tables_list = saved_json['selected_tables_list']
                self.selected_mappings = saved_json['selected_mappings']
                self.source_mapping = saved_json['source_mapping']
                self.source_tables = saved_json['source_tables']
                self.source_profile = saved_json['source_profile']  
                self.SourceDataSet.setCurrentText(saved_json['source_db'])
                self.TargetDataSet.setCurrentText(saved_json['target_model'])
                self.generated_SQL = saved_json['generated_SQL'] 
                self.selected_phi_method = saved_json['phi_result']   
                self.universal_ID = saved_json['universal_id'] 
                self.text_folder = saved_json['text_folder']
                self.label_folder_selected.setText(self.text_folder)
                # todo - display selected tables  / display seleted mappings
                self.update_selected_tables()
                self.update_mapped_tables()
 
 
    # action called by save as action
    def file_saveas(self):
 
        # opening path
        path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Save file", "",
                              "Json File (*.json)")
 
        # if dialog is cancelled i.e no path is selected
        if not path:
            # return this method
            # i.e no action performed
            return
 
        # else call save to path method
        self._save_to_path(path)
 
    # save to path method
    def _save_to_path(self, path):
 
        # get the text
        save_to_json={}
        save_to_json['selected_tables_list']= self.selected_tables_list
        save_to_json['selected_mappings'] = self.selected_mappings
        save_to_json['source_mapping'] = self.source_mapping  
        save_to_json['source_tables'] = self.source_tables 
        save_to_json['source_db'] = self.SourceDataSet.currentText() 
        save_to_json['target_model']= self.TargetDataSet.currentText() 
        save_to_json['source_profile']  = self.source_profile  
        save_to_json['generated_SQL']  = self.generated_SQL
        save_to_json['phi_result']  = self.selected_phi_method  
        save_to_json['universal_id']  =  self.universal_ID 
        save_to_json['text_folder']= self.text_folder        

        # try catch block
        try:
 
            # opening file to write
            with open(path, 'w') as f:
                json.dump(save_to_json, f, indent = 4)
 
        # if error occurs
        except Exception as e:
            # show error using critical
            self.dialog_critical(str(e))
 
    def dialog_critical(self, s):
 
        # creating a QMessageBox object
        dlg = QtWidgets.QMessageBox(self)
 
        # setting text to the dlg
        dlg.setText(s)
 
        # setting icon to it
        dlg.setIcon(QtWidgets.QMessageBox.Critical)
 
        # showing it
        dlg.show()
 

    def update_mapped_tables (self):
        self.mapped_tables_list = []
        for x in self.selected_mappings:
            if [x[0][0][0],x[1][0][0]] not in self.mapped_tables_list:
                self.mapped_tables_list.append([x[0][0][0],x[1][0][0]])

        self.mappedtableModel = QtGui.QStandardItemModel()
        self.mappedtableModel.removeRows( 0, self.mappedtableModel.rowCount() )
        self.mappedtableModel.setHorizontalHeaderLabels(['Mapped Source Tables','Mapped Target Tables']) 
        for mapped_table in self.mapped_tables_list:
            item = [QtGui.QStandardItem(mapped_table[0]),QtGui.QStandardItem(mapped_table[1])]
            self.mappedtableModel.appendRow(item)
        
        self.mapped_tables.setModel(self.mappedtableModel)        
        self.mapped_tables.resizeColumnsToContents()

    def update_selected_tables(self):
        self.selectedtableModel = QtGui.QStandardItemModel()
        self.selectedtableModel.removeRows( 0, self.selectedtableModel.rowCount() )
        self.selectedtableModel.setHorizontalHeaderLabels(['Selected Source Tables']) 
        for source_table in self.selected_tables_list:
            item = QtGui.QStandardItem(source_table)
            self.selectedtableModel.appendRow(item)
        
        self.selected_tables.setModel(self.selectedtableModel)        
        self.selected_tables.resizeColumnsToContents()
    
    def open_window_select_table(self):
        # Open second window for table selection
        from db_tools import get_tables    
        if self.text_folder is None and "TEXT" in self.SourceDataSet.currentText():
            print('folder not selected')
            return
        else:    
            self.source_tables=get_tables(self.SourceDataSet.currentText(),self.text_folder)
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_SelectTableToProcess()
        self.ui.setupUi(self.window,MainWindow=self, source_tables=self.source_tables) # ,selected_tables=self.select_tables
        self.window.show()



    def view_json(self):
 
        self.json_ui = Ui_Json_Viewer()  
        self.jspn_form = QtWidgets.QWidget()
        print(self.source_profile)
        self.json_ui.setupUi( self.jspn_form,source_profile = self.source_profile) # self.source_profile)
        self.jspn_form.show()
  
    



    def gen_db_profile(self):

        from db_tools import get_table_profile
    
        self.source_mapping = {}
        self.source_profile ={}

        for profile_table in self.selected_tables_list:
            print(profile_table)
            table_mapping ={}
            if "TEXT" in self.SourceDataSet.currentText():
                print(os.path.join(self.text_folder,profile_table))
                profile_table_result = get_table_profile(self.SourceDataSet.currentText(),profile_table,data_profile_sample_size,self.text_folder)
            else:                
                profile_table_result = get_table_profile(self.SourceDataSet.currentText(),profile_table,data_profile_sample_size)
            self.source_profile[profile_table] = profile_table_result['data_stats']
            for x in profile_table_result['data_stats']:
                if x['categorical'] :
                    table_mapping[x['column_name']]= [x for  x in x['statistics']['categorical_count']] 
                else:    
                    table_mapping[x['column_name']]=[]
            self.source_mapping[profile_table]= table_mapping      
        self.source_profile =  json.loads(json.dumps(self.source_profile))   
        #with open('data_profile.json', 'w') as f:
        #     json.dump(self.source_profile, f, indent = 4)
        # with open('mapping_profile.json', 'w') as f:
        #     json.dump(self.source_mapping, f, indent = 4)        

    def scan_phi(self):

        phi_scan_result_json={}
        
        self.Profile_logs.clear()
        self.Profile_logs.appendPlainText('PHI Scanning... ')

        for phi_scan_table in self.selected_tables_list:
            original_data_path = './data_profile/table_{}_sample.csv'.format(phi_scan_table)
            json_file_path = './data_profile/table_{}_profile.json'.format(phi_scan_table)
            model_path = PHI_SCAN_MODEL
            output_path = './data_profile/table_{}_phi.csv'.format(phi_scan_table)
            print(phi_scan_table)
            phi_scan(original_data_path,json_file_path,model_path,output_path)
            phi_scan_result = pd.read_csv(output_path,index_col=0)          
            #phi_scan_result = phi_scan_result[phi_scan_result['ML prediction result 0/1']==1]
            
            for index,row in  phi_scan_result.iterrows():
                phi_scan_result_json['{}.{}'.format(phi_scan_table,index)] = {'method':'0','offset':'','predict_probability':row['ML prediction result'],'predict_result':row['ML prediction result 0/1']}

        self.selected_phi_method =  json.loads(json.dumps(phi_scan_result_json,indent = 4))  
        self.Profile_logs.appendPlainText('PHI Scanning Done')        
        print(self.selected_phi_method)

        
    def view_phi_result(self):
        # Open second window for view phi scan result
        self.window = QtWidgets.QMainWindow()
        self.window.setFixedSize(950,600)
        self.ui = Ui_ViewPHIScan()
        self.ui.setupUi(self.window,MainWindow=self) #  
        self.window.show() 

 
 
    def gen_db_profile_thread(self): 

        back_run_list = self.selected_tables_list
      
        if not self._profile_worker.isRunning() and not self._scan_worker.isRunning():
            self._profile_worker.setItems(back_run_list,self.SourceDataSet.currentText(),data_profile_sample_size,self.text_folder)
            self._profile_worker.start()     

 
    def phi_scan_thread(self): 

        back_run_list = self.selected_tables_list
      
        if not self._scan_worker.isRunning() and not self._profile_worker.isRunning() :
            self._scan_worker.setItems(back_run_list,self.SourceDataSet.currentText())
            self._scan_worker.start()   


    def validate_mapping_func(self):
        print(self.selected_mappings) 
        selected_mappings_json = {}
        selected_mappings_json['mapped_items'] = self.selected_mappings
        # with open('selected_mappings.json', 'w') as f:
        #     json.dump(selected_mappings_json, f, indent = 4)
        self.open_window_read_map_validate()

    def set_mapping(self):
        target_mapping_file = target_models[self.TargetDataSet.currentText()]
        with open(target_mapping_file, 'r') as f:
            target_mapping_load = json.load(f)
            self.target_mapping = {}
            for t in target_mapping_load:
                self.target_mapping[t] = {} 
                for fe in target_mapping_load[t]:
                    valid_data = []
                    if 'valid_concepts' in target_mapping_load[t][fe]:
                        for v in target_mapping_load[t][fe]['valid_concepts']:
                            valid_data.append(v)
                    self.target_mapping[t][fe] = valid_data
        # with open('mapping_profile_target.json', 'w') as f:
        #     json.dump(self.target_mapping, f, indent = 4)   
        self.open_window_map_table()
        # print(json.dumps(self.target_mapping,indent=4))    

    def copy_profile_data(self):
        self.source_mapping = self._profile_worker.source_mapping
        self.source_profile = self._profile_worker.source_profile
        self.toLog('all finished... \n ')           

    def copy_scan_data(self):
        self.selected_phi_method = self._scan_worker.phi_scan_result_json
        self.toLog('all finished... \n ') 

    def toLog(self, txt):
        self.Profile_logs.appendPlainText(txt)

    
    def getDirectory(self):

         response = QtWidgets.QFileDialog.getExistingDirectory(
             None,
             caption='Select a folder'
         )
         self.label_folder_selected.setText(str(response) )
         self.text_folder = str(response) 


    def display_text_folder(self):
        if "TEXT" in self.SourceDataSet.currentText() :
            self.label_folder.setVisible(True)
            self.label_folder_selected.setVisible(True)     
        else:            
            self.label_folder.setVisible(False)
            self.label_folder_selected.setVisible(False)

    def setupUi(self, DataMappingTools):


        DataMappingTools.setObjectName("DataMappingTools")
        DataMappingTools.resize(800, 785)
        self.centralwidget = QtWidgets.QWidget(DataMappingTools)
        self.centralwidget.setObjectName("centralwidget")

        self._profile_worker = DB_Profile_Thread(self.centralwidget)
        self._profile_worker.log.connect(self.toLog)
        self._profile_worker.started.connect(lambda: self.toLog('Profiling start... \n '))
        self._profile_worker.finished.connect(lambda: self.copy_profile_data())



        self._scan_worker = PHI_Scan_Thread(self.centralwidget)
        self._scan_worker.log.connect(self.toLog)
        self._scan_worker.started.connect(lambda: self.toLog('PHI Scan start... \n '))
        self._scan_worker.finished.connect(lambda: self.copy_scan_data())


        self.TargetDataSet = QtWidgets.QComboBox(self.centralwidget)
        self.TargetDataSet.addItems(target_models.keys())
        self.TargetDataSet.setGeometry(QtCore.QRect(740, 40, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.TargetDataSet.setFont(font)
        self.TargetDataSet.setObjectName("TargetDataSet")
        self.TargetDataSet.setVisible(False)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(560, 40, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        self.SourceDataSet = QtWidgets.QComboBox(self.centralwidget)
        self.SourceDataSet.addItems(available_dbs.keys())
        self.SourceDataSet.setGeometry(QtCore.QRect(250, 40, 201, 41))
        self.SourceDataSet.currentTextChanged.connect(self.display_text_folder)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SourceDataSet.setFont(font)
        self.SourceDataSet.setObjectName("SourceDataSet")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 40, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_folder = QtWidgets.QPushButton(self.centralwidget,clicked = lambda: self.getDirectory())
        self.label_folder.setGeometry(QtCore.QRect(20, 100, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_folder.setFont(font)
        self.label_folder.setObjectName("label_folder")
        self.label_folder.setVisible(False)
      

        self.label_folder_selected = QtWidgets.QLabel(self.centralwidget)
        self.label_folder_selected.setGeometry(QtCore.QRect(20, 140, 600, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_folder_selected.setFont(font)
        self.label_folder_selected.setObjectName("label_folder_selected")
        self.label_folder_selected.setVisible(False)

        self.selected_tables = QtWidgets.QTableView(self.centralwidget)
        self.selected_tables.setGeometry(QtCore.QRect(20, 300, 381, 161))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.selected_tables.setFont(font)
        self.selected_tables.setObjectName("selected_tables")
        self.update_selected_tables()


        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 260, 241, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.id_generator = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.generate_universal_ID())
        self.id_generator.setGeometry(QtCore.QRect(20, 640, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.id_generator.setFont(font)
        self.id_generator.setObjectName("id_generator")        

        # two function: gen_db_profile_thread  ( run at background) / gen_db_profile
        self.generate_DB_profile = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.gen_db_profile_thread())
        self.generate_DB_profile.setGeometry(QtCore.QRect(20, 540, 140, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.generate_DB_profile.setFont(font)
        self.generate_DB_profile.setObjectName("generate_DB_profile")


        self.view_DB_Profile = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.view_json())
        self.view_DB_Profile.setGeometry(QtCore.QRect(190, 540, 140, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.view_DB_Profile.setFont(font)
        self.view_DB_Profile.setObjectName("view_DB_Profile")

        # two function: phi_scan_thread  ( run at background) / scan_phi
        # self.phi_scan = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.scan_phi())
        self.phi_scan = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.phi_scan_thread())
        self.phi_scan.setGeometry(QtCore.QRect(370, 540, 140, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.phi_scan.setFont(font)
        self.phi_scan.setObjectName("phi_scan")

        self.phi_scan_result = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.view_phi_result())
        self.phi_scan_result.setGeometry(QtCore.QRect(530, 540, 140, 41))
        font = QtGui.QFont() 
        font.setPointSize(12)
        self.phi_scan_result.setFont(font)
        self.phi_scan_result.setObjectName("phi_scan_result")       

        
        self.set_PHI_strategy = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.open_window_select_phi_methods())
        self.set_PHI_strategy.setGeometry(QtCore.QRect(270, 540, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.set_PHI_strategy.setFont(font)
        self.set_PHI_strategy.setObjectName("set_PHI_strategy")



        self.mapped_tables = QtWidgets.QTableView(self.centralwidget)
        self.mapped_tables.setGeometry(QtCore.QRect(560, 260, 381, 161))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.mapped_tables.setFont(font)
        self.mapped_tables.setObjectName("mapped_tables")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(560, 220, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.update_mapped_tables()
        self.mapped_tables.setVisible(False)

        self.validate_mapping = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.validate_mapping_func())
        self.validate_mapping.setGeometry(QtCore.QRect(560, 460, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.validate_mapping.setFont(font)
        self.validate_mapping.setObjectName("validate_mapping")
       

 


        self.start_mapping = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.set_mapping() )
        self.start_mapping.setGeometry(QtCore.QRect(560, 165, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.start_mapping.setFont(font)
        self.start_mapping.setObjectName("start_mapping")

        self.select_tables = QtWidgets.QPushButton(self.centralwidget, clicked = lambda: self.open_window_select_table())
        self.select_tables.setGeometry(QtCore.QRect(20, 195, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.select_tables.setFont(font)
        self.select_tables.setObjectName("select_tables")


        self.generate_SQL = QtWidgets.QPushButton(self.centralwidget,clicked = lambda: self.open_window_gen_SQL())
        self.generate_SQL.setGeometry(QtCore.QRect(560, 540, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.generate_SQL.setFont(font)
        self.generate_SQL.setObjectName("generate_SQL")
        DataMappingTools.setCentralWidget(self.centralwidget)

        self.test_run = QtWidgets.QPushButton(self.centralwidget,clicked = lambda: self.gen_db_profile_thread())
        self.test_run.setGeometry(QtCore.QRect(800, 540, 221, 41))
        self.test_run.hide()

        self.Profile_logs = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.Profile_logs.setGeometry(QtCore.QRect(20, 620, 650,101))
        # logging.getLogger().addHandler(self.SQL_logs)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.Profile_logs.setFont(font)
        self.Profile_logs.setReadOnly(True)

        self.menubar = QtWidgets.QMenuBar(DataMappingTools)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.menubar.setFont(font)
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        DataMappingTools.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(DataMappingTools)
        self.statusbar.setObjectName("statusbar")
        DataMappingTools.setStatusBar(self.statusbar)
        self.actionNew_Project = QtGui.QAction(DataMappingTools)
        self.actionNew_Project.setObjectName("actionNew_Project")
        self.actionNew_Project.triggered.connect(self.new_project)
        self.actionLocal_Project = QtGui.QAction(DataMappingTools)
        self.actionLocal_Project.setObjectName("actionLocal_Project")
        self.actionLocal_Project.triggered.connect(self.file_open)
        self.actionSave_Project = QtGui.QAction(DataMappingTools)
        self.actionSave_Project.setObjectName("actionSave_Project")
        self.actionSave_Project.triggered.connect(self.file_saveas)
        self.actionExit = QtGui.QAction(DataMappingTools)
        self.actionExit.setObjectName("actionExit")
        self.actionData_Profiling = QtGui.QAction(DataMappingTools)
        self.actionData_Profiling.setObjectName("actionData_Profiling")
        self.actionTable_Mapping = QtGui.QAction(DataMappingTools)
        self.actionTable_Mapping.setObjectName("actionTable_Mapping")
        self.actionFiled_Mapping = QtGui.QAction(DataMappingTools)
        self.actionFiled_Mapping.setObjectName("actionFiled_Mapping")
        self.actionSQL_Generator = QtGui.QAction(DataMappingTools)
        self.actionSQL_Generator.setObjectName("actionSQL_Generator")
        self.actionTarget_Data_Standard = QtGui.QAction(DataMappingTools)
        self.actionTarget_Data_Standard.setObjectName("actionTarget_Data_Standard")
        self.actionSource_Data_Selector = QtGui.QAction(DataMappingTools)
        self.actionSource_Data_Selector.setObjectName("actionSource_Data_Selector")
        self.menuFile.addAction(self.actionNew_Project)
        self.menuFile.addAction(self.actionLocal_Project)
        self.menuFile.addAction(self.actionSave_Project)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(DataMappingTools)
        QtCore.QMetaObject.connectSlotsByName(DataMappingTools)

    def retranslateUi(self, DataMappingTools):
        _translate = QtCore.QCoreApplication.translate
        DataMappingTools.setWindowTitle(_translate("DataMappingTools", "ETL Data Mapping Tools"))
        # self.label.setText(_translate("DataMappingTools", "Target Data Model"))
        self.label_2.setText(_translate("DataMappingTools", "Source Database"))
        self.label_folder.setText(_translate("DataMappingTools", "Source Text Files"))
        self.label_folder_selected.setText(_translate("DataMappingTools", ""))
        self.label_3.setText(_translate("DataMappingTools", "Selected Source Tables"))
        self.view_DB_Profile.setText(_translate("DataMappingTools", "View DB Profile"))
        #self.set_PHI_strategy.setText(_translate("DataMappingTools", "Set PHI handle method")) 
        #self.id_generator.setText(_translate("DataMappingTools", "Universal ID Generator"))         
        #self.test_run.setText(_translate("DataMappingTools", "test only - place holder"))         
        #self.label_4.setText(_translate("DataMappingTools", "Mapped Tables"))
        #self.validate_mapping.setText(_translate("DataMappingTools", "Validate Mapping"))
        self.generate_DB_profile.setText(_translate("DataMappingTools", "Genereate \n DB Profile"))
        self.phi_scan.setText(_translate("DataMappingTools", "Scan PHI"))   
        self.phi_scan_result.setText(_translate("DataMappingTools", "View Result"))        
        #self.start_mapping.setText(_translate("DataMappingTools", "Map Data"))
        self.select_tables.setText(_translate("DataMappingTools", "Select Tables to Process"))
        #self.generate_SQL.setText(_translate("DataMappingTools", "Genereate SQL Scripts "))
        self.menuFile.setTitle(_translate("DataMappingTools", "File"))
        self.actionNew_Project.setText(_translate("DataMappingTools", "New Project"))
        self.actionLocal_Project.setText(_translate("DataMappingTools", "Load Project"))
        self.actionSave_Project.setText(_translate("DataMappingTools", "Save Project"))
        self.actionExit.setText(_translate("DataMappingTools", "Exit"))
        self.actionData_Profiling.setText(_translate("DataMappingTools", "Data Profiling"))
        #self.actionTable_Mapping.setText(_translate("DataMappingTools", "Table Mapping"))
        #self.actionFiled_Mapping.setText(_translate("DataMappingTools", "Filed Mapping"))
        #self.actionSQL_Generator.setText(_translate("DataMappingTools", "SQL Generator"))
        #self.actionTarget_Data_Standard.setText(_translate("DataMappingTools", "Target Data Standard"))
        #self.actionSource_Data_Selector.setText(_translate("DataMappingTools", "Source Data Selector"))

        self.label.setVisible(False)
        self.set_PHI_strategy.setVisible(False)
        self.id_generator.setVisible(False)
        self.test_run.setVisible(False)
        self.label_4.setVisible(False)
        self.validate_mapping.setVisible(False)
        self.start_mapping.setVisible(False)
        self.generate_SQL.setVisible(False)
        self.actionSQL_Generator.setVisible(False)
        self.actionTarget_Data_Standard.setVisible(False)
        self.actionSource_Data_Selector.setVisible(False)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DataMappingTools = QtWidgets.QMainWindow()
    ui = Ui_DataMappingTools()
    ui.setupUi(DataMappingTools)
    DataMappingTools.show()
    sys.exit(app.exec())
