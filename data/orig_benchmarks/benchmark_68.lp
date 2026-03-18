d0(o5).
d0(o0).
d2(o5).
d3(o3).
d4(o4).
d1(o1).
d2(o2).
d1(o3).
d4(o3).

d3(X) :- d0(X), d4(X).
d0(X) :- d1(X), d4(X).
d0(X) :- d2(X), d5(X).
d5(X) :- d3(X), d1(X).
d4(X) :- d5(X), d1(X).
d0(X) :- d2(X), d5(X).
d3(X) :- d2(X), d5(X).
d0(X) :- d3(X), d1(X).
d1(X) :- d3(X), d4(X).
d5(X) :- d3(X), d4(X).
d2(X) :- d5(X), d1(X).