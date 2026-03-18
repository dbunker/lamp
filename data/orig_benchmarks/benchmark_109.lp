d1(o0).
d4(o3).
d2(o0).
d5(o4).
d1(o2).
d0(o3).

d1(X) :- d4(X), d5(X).
d4(X) :- d2(X), d0(X).
d4(X) :- d2(X), d3(X).
d0(X) :- d1(X), d3(X).
d4(X) :- d0(X), d3(X).
d3(X) :- d4(X), d0(X).
d2(X) :- d4(X), d0(X).
d5(X) :- d2(X), d1(X).
d3(X) :- d4(X), d5(X).
d2(X) :- d4(X), d1(X).
d4(X) :- d1(X), d3(X).