import os


def read_and_write_py_files(output_file='python_files_content.txt'):
    files = os.listdir()

    py_files = [f for f in files if f.endswith('.py') and f != os.path.basename(__file__)]

    if not py_files:
        print("Не знайдено жодного .py файла в поточній директорії (крім цього скрипта).")
        return

    # Відкриваємо вихідний файл для запису
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for py_file in py_files:
            # Записуємо назву файлу
            out_file.write(f"файл {py_file}\n")

            # Читаємо та записуємо вміст файлу
            try:
                with open(py_file, 'r', encoding='utf-8') as in_file:
                    content = in_file.read()
                    out_file.write(content)

                out_file.write("\n\n")
            except Exception as e:
                out_file.write(f"!!! Помилка при читанні файлу: {str(e)}\n\n")

    print(f"Зміст {len(py_files)} .py файлів було записано у файл {output_file}")

read_and_write_py_files()