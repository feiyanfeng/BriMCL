import functools
import os
import logging

class Logger:
    def __init__(self, filename='log.txt'):
         # 检查输出文件所在的文件夹是否存在，如果不存在则创建它
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        self.logger = logging.getLogger('custom_logger')
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(filename)
        self.handler.setLevel(logging.INFO)
        self.formatters = {0:logging.Formatter('%(message)s'),
                           1:logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')}
        self.formatter = self.formatters[1]
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def get_format(self, format):
        return self.formatters[format] if isinstance(format, int) else logging.Formatter(format)
    def set_format(self, format):
        self.formatter = self.get_format(format)
        self.handler.setFormatter(self.formatter)
    def reset_format(self):
        self.formatter = self.formatters[1]
        self.handler.setFormatter(self.formatter)

    def info(self, message, format=None):
        if format is not None:
            self.handler.setFormatter(self.get_format(format))
        self.logger.info(message)
        if format is not None:
            self.handler.setFormatter(self.formatter)


    def error(self, message, format=None):
        if format is not None:
            self.handler.setFormatter(self.get_format(format))
        self.logger.error(message)
        if format is not None:
            self.handler.setFormatter(self.formatter)

    
    def warning(self, message, format=None):
        if format is not None:
            self.handler.setFormatter(self.get_format(format))
        self.logger.warning(message)
        if format is not None:
            self.handler.setFormatter(self.formatter)


    def critical(self, message, format=None):
        if format is not None:
            self.handler.setFormatter(self.get_format(format))
        self.logger.critical(message)
        if format is not None:
            self.handler.setFormatter(self.formatter)

            
    def close(self):
        # self.logger.info('\n\n\n')
        # self.handler.close()
        logging.shutdown()

    def __del__(self):
        # self.info('end...\n\n\n', '%(message)s')
        self.close()

# if __name__ == '__main__':
    
#     # 使用示例
#     logger = Logger('log/my_log.log')

#     logger.info('This is an info log message1.', 0)
#     logger.info('This is an info log message2.', 1)
# logger.error('This is an error log message.', '%(levelname)s - %(message)s')
# logger.warning('This is a warning log message.')
# logger.critical('This is a critical log message.\n\n\n')


# 程序结束时会自动关闭文件?


'''
定义一个Logger类, 实现: 
1. 调用logger=Logger()时能指定输出到那个文件
2. 如果输日志文件在一个文件夹内，则判断该文件见是否存在，若不存在则创建它
3. 调用logger.info(message, format)时能将信息输出到文件并指定输出格式
4. 同样方式定义logger.error等
5. 程序结束时自动关闭文件
'''

# 将类函数包装，改变函数里面的日志输出格式
# 仅支持类函数；声明Logger类的变量必须是self.logger
# 不支持多线程，不确定会出什么问题
def set_log_format(format):  # 高阶装饰器工厂函数，可以传入所需的值
    def logger_format(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.logger.set_format(format)        
            # 调用原始函数并获取结果
            result = func(self, *args, **kwargs)
            # 恢复logger
            self.logger.reset_format() 
            # 返回结果
            return result
        
        return wrapper
    return logger_format
'''
当你使用装饰器来修改一个函数的行为时，Python 会用装饰器返回的新函数替换掉原函数。然而，这样做的一个副作用是，新函数不再拥有原函数的元数据。这可能会导致一些问题，比如在打印函数名、使用 help() 函数查看文档、或者利用反射机制（如 getattr()）获取函数信息时，得到的是装饰器内部函数（如 wrapper）的信息，而不是原函数的信息。

@functools.wraps(func) 的作用就是解决这个问题。它会更新 wrapper 函数的一些属性，使其看起来更像 func 函数。这样，当你查看 wrapper 函数的元数据时，就会看到原函数 func 的信息，而不是 wrapper 函数自己的信息
'''