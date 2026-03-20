%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.221s
%% Conflict analysis                       : 8.501s
%%   - Positive Examples                   : 8.501s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 2.75s
%% Total                                   : 12.968s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d6(V1) :- d1(V1).
d3(V1) :- d9(V1).
d7(V1) :- d0(V1).
d9(V1) :- d6(V1).
d0(V1) :- d4(V1).

