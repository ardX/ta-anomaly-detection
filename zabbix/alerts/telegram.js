var Telegram = {
  token: null,
  to: null,
  message: null,
  proxy: null,
  parse_mode: null,
  escapeMarkup: function (str, mode) {
      switch (mode) {
          case 'markdown':
              return str.replace(/([_*\[`])/g, '\\$&');
          case 'markdownv2':
              return str.replace(/([_*\[\]()~`>#+\-=|{}.!])/g, '\\$&');
          case 'html':
              return str.replace(/<(\s|[^a-z\/])/g, '&lt;$1');
          default:
              return str;
      }
  },
  sendMessage: function () {
      var params = {
          chat_id: Telegram.to,
          text: Telegram.message,
          disable_web_page_preview: true,
          disable_notification: false
      },
      data,
      response,
      request = new HttpRequest(),
      url = 'https://api.telegram.org/bot' + Telegram.token + '/sendMessage';
      if (Telegram.parse_mode !== null) {
          params['parse_mode'] = Telegram.parse_mode;
      }
      if (Telegram.proxy) {
          request.setProxy(Telegram.proxy);
      }
      request.addHeader('Content-Type: application/json');
      data = JSON.stringify(params);
      // Remove replace() function if you want to see the exposed token in the log file.
      Zabbix.log(4, '[Telegram Webhook] URL: ' + url.replace(Telegram.token, '<TOKEN>'));
      Zabbix.log(4, '[Telegram Webhook] params: ' + data);
      response = request.post(url, data);
      Zabbix.log(4, '[Telegram Webhook] HTTP code: ' + request.getStatus());
      try {
          response = JSON.parse(response);
      }
      catch (error) {
          response = null;
      }
      if (request.getStatus() !== 200 || typeof response.ok !== 'boolean' || response.ok !== true) {
          if (typeof response.description === 'string') {
              throw response.description;
          }
          else {
              throw 'Unknown error. Check debug log for more information.';
          }
      }
  }
};

// Function to format CSV message into readable text
function formatTelegramMessage(csvData) {
  // Define headers
  var headers = [
      "timestamp",
      "cpu_usage",
      "top_1_cpu_proc_name",
      "top_1_cpu_proc_usage",
      "top_2_cpu_proc_name",
      "top_2_cpu_proc_usage",
      "top_3_cpu_proc_name",
      "top_3_cpu_proc_usage",
      "top_4_cpu_proc_name",
      "top_4_cpu_proc_usage",
      "top_5_cpu_proc_name",
      "top_5_cpu_proc_usage",
      "mem_usage",
      "top_1_mem_proc_name",
      "top_1_mem_proc_usage",
      "top_2_mem_proc_name",
      "top_2_mem_proc_usage",
      "top_3_mem_proc_name",
      "top_3_mem_proc_usage",
      "top_4_mem_proc_name",
      "top_4_mem_proc_usage",
      "top_5_mem_proc_name",
      "top_5_mem_proc_usage",
      "nginx_active_connections",
      "nginx_rps",
      "is_anomaly"
  ];
  
  // Split the CSV data by semicolons
  var values = csvData.split(';');
  var message = "";
  
  // Create a formatted message
  for (var i = 0; i < Math.min(headers.length, values.length); i++) {
      // Format the header with proper capitalization and spaces
      var headerParts = headers[i].split('_');
      var formattedHeader = "";
      
      for (var j = 0; j < headerParts.length; j++) {
          formattedHeader += headerParts[j].charAt(0).toUpperCase() + headerParts[j].slice(1) + " ";
      }
      
      formattedHeader = formattedHeader.trim();
      message += formattedHeader + ": " + values[i] + "\n";
  }
  
  return message;
}

try {
  var params = JSON.parse(value);
  if (typeof params.Token === 'undefined') {
      throw 'Incorrect value is given for parameter "Token": parameter is missing';
  }
  Telegram.token = params.Token;
  if (params.HTTPProxy) {
      Telegram.proxy = params.HTTPProxy;
  } 
  params.ParseMode = params.ParseMode.toLowerCase();
  
  if (['markdown', 'html', 'markdownv2'].indexOf(params.ParseMode) !== -1) {
      Telegram.parse_mode = params.ParseMode;
  }
  Telegram.to = params.To;
  
  // Format the message using the formatTelegramMessage function
  var formattedMessage = formatTelegramMessage(params.Message);
  Telegram.message = params.Subject + '\n' + formattedMessage;
  
  if (['markdown', 'html', 'markdownv2'].indexOf(params.ParseMode) !== -1) {
      Telegram.message = Telegram.escapeMarkup(Telegram.message, params.ParseMode);
  }
  Telegram.sendMessage();
  return 'OK';
}
catch (error) {
  Zabbix.log(4, '[Telegram Webhook] notification failed: ' + error);
  throw 'Sending failed: ' + error + '.';
}