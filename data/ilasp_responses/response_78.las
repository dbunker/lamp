%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.223s
%% Conflict analysis                       : 1.347s
%%   - Positive Examples                   : 1.347s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.552s
%% Total                                   : 2.183s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d5(V1).
d4(V1) :- d3(V1).
d0(V1) :- d3(V1).
d1(V1) :- d0(V1).
d2(V1) :- d3(V1).

