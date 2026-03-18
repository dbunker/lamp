d0(o0).
d1(o1).
d5(o1).
d2(o1).
d3(o5).
d4(o3).

d5(X) :- d0(X).
d4(X) :- d2(X).
d3(X) :- not d2(X), d1(X).
d1(X) :- not d2(X), d0(X).
d0(X) :- not d2(X), d3(X).