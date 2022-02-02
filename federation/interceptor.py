# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS CLIENT                               ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
from grpc_interceptor import ServerInterceptor


##############################################################################
#                                ERRORLOGGER                                 #
##############################################################################
# TODO: Implement!
class ErrorLogger(ServerInterceptor):
    def intercept(self, method, request, context, method_name):
        try:
            return method(request, context)
        except Exception as e:
            self.log_error(e)
            raise

    #def log_error(self, e: Exception) -> None: