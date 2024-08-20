    elif config.train_mode == 'meta_sg':
        for i in progress_bar:
            
            input, target = next(dataloader.__iter__())

            k_shot = config.meta_k #task_number
            support_c_num = config.n_classes
            
            update_lr = 0.01
            update_step = 2

            losses_q = [0 for _ in range(update_step + 1)]  # losses_q[i] is the loss on step i

            for _ in range(0, k_shot):
                for c_i in [x for x in range(1, support_c_num) if x != config.t_class]:
                    support_image = input
                    support_label = []
                    query_label = []

                    copy_support_bg = target[:,c_i,:,:,:]-1.
                    copy_support_bg = abs(copy_support_bg)
                    support_label.append(copy_support_bg)
                    support_label.append(target[:,c_i,:,:,:])
                    support_label = torch.cat(support_label, dim=0).unsqueeze(0)

                    query_image = input
                    copy_query_bg = target[:,config.t_class,:,:,:]-1.
                    copy_query_bg = abs(copy_query_bg)
                    query_label.append(copy_query_bg)
                    query_label.append(target[:,config.t_class,:,:,:])
                    query_label = torch.cat(query_label, dim=0).unsqueeze(0)

                    if torch.cuda.is_available():
                        support_image =  support_image.cuda()
                        query_image =  support_image # save GPU memory

                        support_label = support_label.cuda()
                        query_label = query_label.cuda()
            
                    # 1. run the i-th task and compute loss for k=0
                    logits = net_S(support_image,vars=None)
                    loss = loss_S(logits, support_label)
                    grad = torch.autograd.grad(loss, net_S.parameters())
                    fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, net_S.parameters())))   
                    
                    # fast_weights = [torch.nn.Parameter(p) for p in fast_weights]
                    
                    for k in range(1, update_step):
                        # 1. run the i-th task and compute loss for k=1~K-1
                        logits = net_S(support_image, fast_weights)
                        loss = loss_S(logits, support_label)

                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, fast_weights)

                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, fast_weights)))

                        # fast_weights = [torch.nn.Parameter(p) for p in fast_weights]

                        logits_q = net_S(query_image,fast_weights)
                        # loss_q will be overwritten and just keep the loss_q on last update step.
                        loss_q = loss_S(logits_q, query_label)
                        losses_q[k + 1] += loss_q

            loss_q = losses_q[-1] / k_shot
            loss_q.backward()
            opt_S.step()
            net_S.zero_grad()
            opt_S.zero_grad()
            loss_S_log.update(loss.data, target.size(0))

            if online=='True':
                wandb.log({'Loss_S': (loss_S_log.avg)})
                
            progress_bar.set_description(f"Epoch: {epoch}/{n_epochs}-{i}")
            progress_bar.set_postfix_str(f"Loss: {loss_S_log.avg:.4f}")
    