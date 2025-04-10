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
          text: Telegram.message.split(";"),
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
  Telegram.message = params.Subject + '\n' + params.Message;

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