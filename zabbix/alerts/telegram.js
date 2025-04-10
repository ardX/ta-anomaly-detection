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

function formatTelegramMessage(csvData) {
  try {
      var values = csvData.split(';');
      var message = "";
      
      var timestamp = values[0];
      var date = new Date(timestamp);

      message += date.toLocaleDateString()+ " " + date.toLocaleTimeString() + "\n\n";
      
      message += "Penggunaan CPU: " + values[1] + "%\n";
      message += "Penggunaan Memory: " + values[12] + "%\n\n";
      
      message += "Penggunaan Proses Tertinggi pada CPU:\n";
      for (var i = 1; i <= 5; i++) {
          var procNameIndex = 2 + (i-1)*2;
          var procUsageIndex = 3 + (i-1)*2;
          if (procNameIndex < values.length && procUsageIndex < values.length) {
              message += i + ". " + values[procNameIndex] + " = " + values[procUsageIndex] + "%\n";
          }
      }
      
      message += "\n";
      
      message += "Penggunaan Proses Tertinggi pada Memory:\n";
      for (var i = 1; i <= 5; i++) {
          var procNameIndex = 13 + (i-1)*2;
          var procUsageIndex = 14 + (i-1)*2;
          if (procNameIndex < values.length && procUsageIndex < values.length) {
              message += i + ". " + values[procNameIndex] + " = " + values[procUsageIndex] + "%\n";
          }
      }
      
      message += "\n";
      
      var nginxConnections = values[23] || "0";
      var nginxRps = values[24] || "0";
      message += "Jumlah koneksi aktif pada webserver: " + nginxConnections + "\n";
      message += "Jumlah permintaan per detik di webserver: " + nginxRps;
      
      return message;
  } catch (error) {
      Zabbix.log(3, '[Telegram Webhook] Error formatting message: ' + error);
      return "Error formatting message. Check Zabbix logs for details.";
  }
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
  
  if (typeof params.ParseMode !== 'undefined') {
      params.ParseMode = params.ParseMode.toLowerCase();
      if (['markdown', 'html', 'markdownv2'].indexOf(params.ParseMode) !== -1) {
          Telegram.parse_mode = params.ParseMode;
      }
  }
  
  Telegram.to = params.To;
  
  var formattedMessage = formatTelegramMessage(params.Message);
  Telegram.message = formattedMessage;
  
  if (Telegram.parse_mode !== null) {
      Telegram.message = Telegram.escapeMarkup(Telegram.message, Telegram.parse_mode);
  }
  
  Telegram.sendMessage();
  return 'OK';
}
catch (error) {
  Zabbix.log(4, '[Telegram Webhook] notification failed: ' + error);
  throw 'Sending failed: ' + error + '.';
}