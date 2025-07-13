import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from pathlib import Path

class PersonaManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("人设管理工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 人设数据
        self.personas = {}
        self.personas_file = "personas.json"
        
        # 创建界面
        self.create_widgets()
        
        # 加载数据
        self.load_personas()
        
        # 刷新列表
        self.refresh_persona_list()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置根窗口的行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="人设管理工具", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 左侧：人设列表
        list_frame = ttk.LabelFrame(main_frame, text="人设列表", padding="5")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # 人设列表(Treeview)
        columns = ("名称", "触发词", "描述", "默认")
        self.persona_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列标题和宽度
        self.persona_tree.heading("名称", text="名称")
        self.persona_tree.heading("触发词", text="触发词")
        self.persona_tree.heading("描述", text="描述")
        self.persona_tree.heading("默认", text="默认")
        
        self.persona_tree.column("名称", width=100)
        self.persona_tree.column("触发词", width=80)
        self.persona_tree.column("描述", width=200)
        self.persona_tree.column("默认", width=50)
        
        # 滚动条
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.persona_tree.yview)
        self.persona_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.persona_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 绑定选择事件
        self.persona_tree.bind("<<TreeviewSelect>>", self.on_select_persona)
        
        # 右侧：编辑区域
        edit_frame = ttk.LabelFrame(main_frame, text="编辑人设", padding="5")
        edit_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        edit_frame.columnconfigure(1, weight=1)
        
        # 人设名称
        ttk.Label(edit_frame, text="人设名称:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(edit_frame, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # 文件名
        ttk.Label(edit_frame, text="文件名:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(edit_frame, textvariable=self.file_var, width=30)
        self.file_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # 触发词
        ttk.Label(edit_frame, text="触发词:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.keywords_var = tk.StringVar()
        self.keywords_entry = ttk.Entry(edit_frame, textvariable=self.keywords_var, width=30)
        self.keywords_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # 描述
        ttk.Label(edit_frame, text="描述:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.description_var = tk.StringVar()
        self.description_entry = ttk.Entry(edit_frame, textvariable=self.description_var, width=30)
        self.description_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # 是否默认
        self.is_default_var = tk.BooleanVar()
        self.default_check = ttk.Checkbutton(edit_frame, text="设为默认人设", variable=self.is_default_var)
        self.default_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 人设内容
        ttk.Label(edit_frame, text="人设内容:").grid(row=5, column=0, sticky=tk.W, pady=2)
        content_frame = ttk.Frame(edit_frame)
        content_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        edit_frame.rowconfigure(6, weight=1)
        
        self.content_text = tk.Text(content_frame, height=10, width=40, wrap=tk.WORD)
        content_scroll = ttk.Scrollbar(content_frame, orient="vertical", command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=content_scroll.set)
        
        self.content_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 按钮区域
        button_frame = ttk.Frame(edit_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="新增", command=self.add_persona).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="保存", command=self.save_persona).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="删除", command=self.delete_persona).grid(row=0, column=2, padx=2)
        ttk.Button(button_frame, text="清空", command=self.clear_form).grid(row=0, column=3, padx=2)
        
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(bottom_frame, text="导入配置", command=self.import_config).grid(row=0, column=0, padx=5)
        ttk.Button(bottom_frame, text="导出配置", command=self.export_config).grid(row=0, column=1, padx=5)
        ttk.Button(bottom_frame, text="刷新", command=self.refresh_persona_list).grid(row=0, column=2, padx=5)
        
        # 存储当前编辑的人设名称
        self.current_editing = None
    
    def load_personas(self):
        """加载人设配置"""
        try:
            if os.path.exists(self.personas_file):
                with open(self.personas_file, 'r', encoding='utf-8') as f:
                    self.personas = json.load(f)
            else:
                # 创建默认配置
                self.personas = {
                    "默认助手": {
                        "file": "default.txt",
                        "keywords": "默认",
                        "is_default": True,
                        "description": "友善专业的AI助手"
                    }
                }
                self.save_config()
        except Exception as e:
            messagebox.showerror("错误", f"加载配置失败: {str(e)}")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.personas_file, 'w', encoding='utf-8') as f:
                json.dump(self.personas, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")
    
    def refresh_persona_list(self):
        """刷新人设列表"""
        # 清空列表
        for item in self.persona_tree.get_children():
            self.persona_tree.delete(item)
        
        # 添加人设到列表
        for name, persona in self.personas.items():
            default_text = "是" if persona.get("is_default", False) else "否"
            self.persona_tree.insert("", "end", values=(
                name,
                persona.get("keywords", ""),
                persona.get("description", ""),
                default_text
            ))
    
    def on_select_persona(self, event):
        """选择人设时的处理"""
        selection = self.persona_tree.selection()
        if selection:
            item = self.persona_tree.item(selection[0])
            persona_name = item["values"][0]
            self.load_persona_to_form(persona_name)
    
    def load_persona_to_form(self, persona_name):
        """加载人设到表单"""
        if persona_name in self.personas:
            persona = self.personas[persona_name]
            
            self.name_var.set(persona_name)
            self.file_var.set(persona.get("file", ""))
            self.keywords_var.set(persona.get("keywords", ""))
            self.description_var.set(persona.get("description", ""))
            self.is_default_var.set(persona.get("is_default", False))
            
            # 加载人设内容
            self.content_text.delete("1.0", tk.END)
            content = self.load_persona_content(persona.get("file", ""))
            self.content_text.insert("1.0", content)
            
            self.current_editing = persona_name
    
    def load_persona_content(self, filename):
        """加载人设内容文件"""
        if not filename:
            return ""
        
        personas_dir = Path("Hyperbot/plugins/deepseek/personas")
        file_path = personas_dir / filename
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            messagebox.showwarning("警告", f"无法读取文件 {filename}: {str(e)}")
        
        return ""
    
    def save_persona_content(self, filename, content):
        """保存人设内容到文件"""
        if not filename:
            return
        
        personas_dir = Path("Hyperbot/plugins/deepseek/personas")
        personas_dir.mkdir(parents=True, exist_ok=True)
        file_path = personas_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("错误", f"保存文件 {filename} 失败: {str(e)}")
    
    def add_persona(self):
        """添加新人设"""
        self.clear_form()
        self.current_editing = None
        self.name_entry.focus()
    
    def save_persona(self):
        """保存人设"""
        name = self.name_var.get().strip()
        file_name = self.file_var.get().strip()
        keywords = self.keywords_var.get().strip()
        description = self.description_var.get().strip()
        is_default = self.is_default_var.get()
        content = self.content_text.get("1.0", tk.END).strip()
        
        if not name:
            messagebox.showerror("错误", "请输入人设名称")
            return
        
        if not keywords:
            messagebox.showerror("错误", "请输入触发词")
            return
        
        if not description:
            messagebox.showerror("错误", "请输入人设描述")
            return
        
        # 如果没有指定文件名，自动生成
        if not file_name:
            file_name = f"{name.lower().replace(' ', '_')}.txt"
        
        # 检查是否是新增还是编辑
        if self.current_editing and self.current_editing != name:
            # 如果改了名称，删除旧的
            if self.current_editing in self.personas:
                del self.personas[self.current_editing]
        
        # 如果设为默认，清除其他默认标记
        if is_default:
            for persona_name in self.personas:
                self.personas[persona_name]["is_default"] = False
        
        # 保存人设配置
        self.personas[name] = {
            "file": file_name,
            "keywords": keywords,
            "is_default": is_default,
            "description": description
        }
        
        # 保存人设内容到文件
        self.save_persona_content(file_name, content)
        
        # 保存配置
        self.save_config()
        
        # 刷新列表
        self.refresh_persona_list()
        
        # 更新当前编辑状态
        self.current_editing = name
        
        messagebox.showinfo("成功", "人设保存成功")
    
    def delete_persona(self):
        """删除人设"""
        if not self.current_editing:
            messagebox.showwarning("警告", "请先选择要删除的人设")
            return
        
        if self.personas.get(self.current_editing, {}).get("is_default", False):
            messagebox.showwarning("警告", "不能删除默认人设")
            return
        
        if messagebox.askyesno("确认", f"确定要删除人设 '{self.current_editing}' 吗？"):
            if self.current_editing in self.personas:
                del self.personas[self.current_editing]
                self.save_config()
                self.refresh_persona_list()
                self.clear_form()
                messagebox.showinfo("成功", "人设删除成功")
    
    def clear_form(self):
        """清空表单"""
        self.name_var.set("")
        self.file_var.set("")
        self.keywords_var.set("")
        self.description_var.set("")
        self.is_default_var.set(False)
        self.content_text.delete("1.0", tk.END)
        self.current_editing = None
    
    def import_config(self):
        """导入配置"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_personas = json.load(f)
                
                self.personas.update(imported_personas)
                self.save_config()
                self.refresh_persona_list()
                messagebox.showinfo("成功", "配置导入成功")
            except Exception as e:
                messagebox.showerror("错误", f"导入配置失败: {str(e)}")
    
    def export_config(self):
        """导出配置"""
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.personas, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", "配置导出成功")
            except Exception as e:
                messagebox.showerror("错误", f"导出配置失败: {str(e)}")

def main():
    root = tk.Tk()
    app = PersonaManagerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()