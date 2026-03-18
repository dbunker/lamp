d2(o5).
d3(o5).
d5(o5).
d4(o3).
d4(o4).

d0(X) :- d4(X), d2(X), d3(X).
d2(X) :- d1(X), d0(X), d3(X).
d0(X) :- d2(X), d1(X), d4(X).
d0(X) :- d4(X), d1(X), d3(X).
d1(X) :- d2(X), d0(X), d3(X).
d0(X) :- d2(X), d5(X), d3(X).
d2(X) :- d1(X), d0(X), d5(X).
d1(X) :- d0(X), d5(X), d3(X).
d4(X) :- d1(X), d5(X), d3(X).
d5(X) :- d1(X), d0(X), d3(X).
d2(X) :- d4(X), d5(X), d3(X).