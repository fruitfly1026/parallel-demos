/************************************************************
                        Course          :       CSC456
                        Source          :       msi.cc
                        Instructor      :       Ed Gehringer
                        Email Id        :       efg@ncsu.edu
------------------------------------------------------------
        © Please do not replicate this code without consulting
                the owner & instructor! Thanks!
*************************************************************/
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
using namespace std;
#include "main.h"
#include "moesi.h"


/*
------------------------------------------------------
Don't modify the fucntions included in this section
------------------------------------------------------
*/
cache_line * MOESI::allocate_line_dir(ulong addr) {
	return NULL;
}
void MOESI::signalRd(ulong addr, int processor_number){
}
void MOESI::signalRdX(ulong addr, int processor_number){
}
void MOESI::signalUpgr(ulong addr, int processor_number){
}
void MOESI::Inv(ulong addr) {
}
void MOESI::Int(ulong addr) {
}
void MOESI::PrRdDir(ulong addr, int processor_number) {
}
void MOESI::PrWrDir(ulong addr, int processor_number) {
}
//-------------------------------------------------------------
//Section ends here. edit the functions in the section below 
//-------------------------------------------------------------

void MOESI::PrRd(ulong addr, int processor_number) {
        cache_state state;
        reads++;
        current_cycle++;
        cache_line *line = find_line(addr);
        if(line == NULL || line->get_state() == I) {
                read_misses++;
                cache_line *newline = allocate_line(addr);
                bus_reads++;
                int count = sharers_exclude(addr,processor_number);
                if (count) {
                        int c2c = c2c_supplier(addr,processor_number);
                        if(c2c) {
                                cache2cache++;
                        } else {
                                memory_transactions++;
                        }
                        I2S++;
                        newline->set_state(S);
                } else {
                        memory_transactions++;
                        I2E++;        
                        newline->set_state(E);
                }
                sendBusRd(addr, processor_number);
        } else {
                state = line->get_state();
                if(state == E || state == M || state == O || state == S) {
                        update_LRU(line);
                }
        }
}


void MOESI::PrWr(ulong addr, int processor_number) {	
        cache_state state;
        current_cycle++;
        writes++;
        cache_line *line = find_line(addr);
        if(line == NULL || line->get_state() == I) {
                write_misses++;
                cache_line *newline = allocate_line(addr);
                int c2c = c2c_supplier(addr, processor_number);
                if (c2c) {
                        cache2cache++;
                        I2M++;
                        newline->set_state(M);
                        bus_readxs++;
                        sendBusRdX(addr, processor_number);
                } else {
                        memory_transactions++;
                        bus_readxs++;
                        I2M++;
                        sendBusRdX(addr, processor_number);
                        newline->set_state(M);
                }
        } else {
                state = line->get_state();
                if(state == E) {
                        E2M++;
                        line->set_state(M);
                        update_LRU(line);
                } else if(state == M) {
                        update_LRU(line);
                } else if(state == O) {
                        O2M++;
                        line->set_state(M);
                        bus_upgrades++;
                        update_LRU(line);
                        sendBusUpgr(addr, processor_number);
                } else if(state == S) {
                        S2M++;
                        line->set_state(M);
                        update_LRU(line);
                        bus_upgrades++;
                        sendBusUpgr(addr, processor_number);
                }
        }
}


void MOESI::BusRd(ulong addr) {
        cache_state state;
        cache_line * line=find_line(addr);
        if(line != NULL) {
                state = line->get_state();
                if(state == E) {
                        E2S++;
                        line->set_state(S);
                } else if(state == M) {
                        M2O++;
                        interventions++;
                        line->set_state(O);
                        flushes++;
                } else if(state == O) {
                        flushes++;
                }
        }
}

void MOESI::BusRdX(ulong addr) {
        cache_state state;
        cache_line * line=find_line(addr);
        if(line != NULL) {
                state = line->get_state();
                if(state == M) {
                        M2I++;
                        flushes++;
                        invalidations++;
                        line->set_state(I);
                } else if(state == O) {
                        flushes++;
                        invalidations++;
                        line->set_state(I);
                } else if(state == S || state == E) {
                        invalidations++;
                        line->set_state(I);
                }
        }
}


void MOESI::BusUpgr(ulong addr) {
        cache_state state;
        cache_line * line=find_line(addr);
        if (line!=NULL) {
                state = line->get_state();
                if(state == S || state == O) {
                        invalidations++;
                        line->set_state(I);
                }
        }
}


bool MOESI::writeback_needed(cache_state state) {
	if (state == M || state == O) {
                return true;
        } else {
                return false;
        }
}

