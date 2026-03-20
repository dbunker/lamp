d3(o1).
d1(o4).
d0(o5).
d4(o0).
d2(o5).
d5(o4).
d2(o1).
d0(o2).
d3(o4).

d2(X) :- not d0(X), d3(X).
d5(X) :- not d0(X), d2(X), not d1(X).
d4(X) :- not d0(X), d2(X).